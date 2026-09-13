"""데이터 특성 기반 공통 조건 자동 제안.

배치 실험의 공정 비교 조건(공통 K, BERTopic min_cluster_size, QualIT LLM 상한)을
코퍼스 크기·임베딩 구조·데이터 체제에서 추정한다. 전 과정이 결정적(고정 시드)이라
같은 코퍼스에는 항상 같은 제안이 나온다.

알고리즘:
1) 공통 K — 문서 임베딩(대량이면 최대 3,000건 무작위 표본)을 K-means로 k 후보 스캔,
   코사인 실루엣을 계산. 단순 argmax는 뉴스처럼 다양한 코퍼스에서 k=2~3의 굵은 분할로
   붕괴하는 경향이 있어(관측된 문제), **관용 규칙**을 쓴다: 최고 실루엣의 90% 이상인
   k 중 가장 큰 값 — 통계의 1-SE 규칙과 같은 취지로, 품질 손실 없이 가장 정보량 많은
   분할을 고른다. 논문 체제에서는 원문(논문) 수의 2배를 상한으로 추가 제약.
2) BERTopic min_cluster_size — HDBSCAN이 K개 이상을 찾을 수 있도록
   n_docs / (K × 20)을 제안 (기본 n/20은 대량 코퍼스에서 토픽 붕괴 유발).
3) QualIT LLM 상한 — 토픽당 대표 50건 기준 K × 50 (200~1,000 범위, MCPanal의
   토픽당 18건 × 여유율을 단일 코퍼스 규모로 환산).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.types import Corpus

logger = logging.getLogger(__name__)

_RANDOM_STATE = 0            # 제안은 결정적이어야 함 (실험 시드와 무관)
_SAMPLE_SIZE = 3000
_K_CANDIDATES = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]
_TOLERANCE = 0.90            # 관용 규칙: 최고 실루엣 × 이 비율 이상이면 후보 유지


@dataclass
class CommonConditionRecommendation:
    n_topics: int
    bertopic_min_cluster_size: int
    qualit_max_llm_docs: int
    k_scores: Dict[int, float] = field(default_factory=dict)
    rationale: List[str] = field(default_factory=list)


def _scan_k_silhouette(embeddings: np.ndarray, candidates: List[int]) -> Dict[int, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores: Dict[int, float] = {}
    for k in candidates:
        try:
            labels = KMeans(n_clusters=k, random_state=_RANDOM_STATE, n_init=4).fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            scores[k] = float(silhouette_score(embeddings, labels, metric="cosine"))
        except Exception as e:
            logger.warning("k=%d 실루엣 스캔 실패: %s", k, e)
    return scores


def _pick_k(scores: Dict[int, float]) -> int:
    """관용 규칙: 최고 실루엣의 _TOLERANCE 이상인 k 중 최댓값."""
    positive = {k: s for k, s in scores.items() if s > 0}
    if not positive:
        return max(scores, key=scores.get) if scores else 10
    best = max(positive.values())
    eligible = [k for k, s in positive.items() if s >= best * _TOLERANCE]
    return max(eligible)


def recommend_common_conditions(
    corpus: Corpus,
    embeddings: np.ndarray,
    sample_size: int = _SAMPLE_SIZE,
    k_candidates: Optional[List[int]] = None,
) -> CommonConditionRecommendation:
    n_docs = len(corpus)
    rationale: List[str] = [f"코퍼스: {corpus.regime}, {n_docs:,}건"]

    # ---- 표본 추출 (결정적) ----
    matrix = np.asarray(embeddings, dtype=np.float32)
    if n_docs > sample_size:
        rng = np.random.default_rng(_RANDOM_STATE)
        idx = rng.choice(n_docs, size=sample_size, replace=False)
        matrix = matrix[np.sort(idx)]
        rationale.append(f"임베딩 표본 {sample_size:,}건으로 k 스캔 (고정 시드, 결정적)")

    # ---- 1) 공통 K ----
    candidates = [k for k in (k_candidates or _K_CANDIDATES) if k < len(matrix) // 2]
    if corpus.regime == "papers":
        n_sources = len({d.source_id for d in corpus.docs})
        source_cap = max(4, n_sources * 2)
        candidates = [k for k in candidates if k <= source_cap] or [max(2, min(4, source_cap))]
        rationale.append(f"논문 체제: 원문 {n_sources}편 → K 상한 {source_cap} (원문 수 × 2)")

    scores = _scan_k_silhouette(matrix, candidates)
    n_topics = _pick_k(scores) if scores else 10
    if scores:
        best_k = max(scores, key=scores.get)
        rationale.append(
            f"실루엣 스캔(코사인): 최고 k={best_k} ({scores[best_k]:.3f}) → "
            f"관용 규칙(최고의 {int(_TOLERANCE * 100)}% 이상 중 최대 k)으로 K={n_topics} "
            f"({scores.get(n_topics, float('nan')):.3f}) — 굵은 분할 붕괴 방지"
        )
    else:
        rationale.append("실루엣 스캔 실패 → 기본값 K=10")

    # ---- 2) BERTopic min_cluster_size ----
    mcs = int(np.clip(n_docs // (n_topics * 20), 3, 500))
    rationale.append(
        f"BERTopic mcs={mcs}: n/(K×20) — HDBSCAN이 K={n_topics}개 이상 찾도록 "
        f"기본값(n/20={max(3, n_docs // 20)})보다 완화"
    )

    # ---- 3) QualIT LLM 대표 문서 상한 ----
    llm_docs = int(min(n_docs, np.clip(n_topics * 50, 200, 1000)))
    if llm_docs >= n_docs:
        rationale.append(f"QualIT LLM 상한 {llm_docs}: 코퍼스 전체 (소규모라 샘플링 불필요)")
    else:
        rationale.append(f"QualIT LLM 상한 {llm_docs}: 토픽당 대표 ~50건 × K={n_topics} (200~1,000 클램프)")

    return CommonConditionRecommendation(
        n_topics=n_topics,
        bertopic_min_cluster_size=mcs,
        qualit_max_llm_docs=llm_docs,
        k_scores=scores,
        rationale=rationale,
    )
