"""2단계: 환각(hallucination) 필터 — ablation 지점.

- cosine: 키프레이즈-원문 임베딩 코사인 유사도 < 임계값 제거 (QualIT 논문, 기본 0.10)
- percentile: 전체 키프레이즈 코사인 분포의 하위 p% 제거 (개선 후보: 코퍼스 적응형)
- llm_verify: LLM이 원문 근거 여부를 직접 검증 (개선 후보: 의미적 검증)
- none: 필터 없음 (ablation 대조군)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from core.llm.json_utils import parse_json_response
from core.llm.ollama_client import OllamaClient
from core.utils.hashing import text_hash

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_coherences(
    keyphrases_per_doc: List[List[str]],
    doc_embeddings: np.ndarray,
    kp_embedding_lookup: Dict[str, np.ndarray],
) -> List[List[float]]:
    """문서별 각 키프레이즈의 원문 정합성(코사인). 임베딩 실패는 nan."""
    out: List[List[float]] = []
    for doc_idx, kps in enumerate(keyphrases_per_doc):
        row = []
        for kp in kps:
            emb = kp_embedding_lookup.get(kp)
            if emb is None or doc_idx >= len(doc_embeddings):
                row.append(float("nan"))
            else:
                row.append(_cosine(doc_embeddings[doc_idx], emb))
        out.append(row)
    return out


def filter_keyphrases(
    method: str,
    keyphrases_per_doc: List[List[str]],
    doc_embeddings: np.ndarray,
    kp_embedding_lookup: Dict[str, np.ndarray],
    threshold: float = 0.10,
    llm: OllamaClient | None = None,
    llm_model: str = "",
    texts: List[str] | None = None,
) -> Tuple[List[List[str]], Dict[str, float]]:
    """필터링된 키프레이즈와 통계(원본/제거 수, 제거율)를 반환."""
    raw_count = sum(len(kps) for kps in keyphrases_per_doc)

    if method == "none" or raw_count == 0:
        return keyphrases_per_doc, _stats(raw_count, keyphrases_per_doc)

    if method == "llm_verify":
        if llm is None or not llm_model or texts is None:
            raise ValueError("llm_verify 필터에는 llm/llm_model/texts가 필요합니다.")
        filtered = _llm_verify_filter(keyphrases_per_doc, texts, llm, llm_model)
        return filtered, _stats(raw_count, filtered)

    coherences = compute_coherences(keyphrases_per_doc, doc_embeddings, kp_embedding_lookup)

    if method == "percentile":
        # 하위 threshold 비율 제거 (분포 적응형)
        all_values = [c for row in coherences for c in row if not np.isnan(c)]
        if not all_values:
            return keyphrases_per_doc, _stats(raw_count, keyphrases_per_doc)
        cutoff = float(np.quantile(all_values, threshold))
    elif method == "cosine":
        cutoff = threshold
    else:
        raise ValueError(f"알 수 없는 환각 필터: {method}")

    filtered = []
    for kps, cohs in zip(keyphrases_per_doc, coherences):
        # 임베딩 실패(nan)는 과도한 삭제 방지 위해 유지 (MCPanal 정책)
        filtered.append([kp for kp, c in zip(kps, cohs) if np.isnan(c) or c >= cutoff])
    return filtered, _stats(raw_count, filtered)


def _stats(raw_count: int, filtered: List[List[str]]) -> Dict[str, float]:
    kept = sum(len(kps) for kps in filtered)
    removed = max(0, raw_count - kept)
    return {
        "hallucination_raw_keyphrases": float(raw_count),
        "hallucination_kept_keyphrases": float(kept),
        "hallucination_removed_keyphrases": float(removed),
        "hallucination_removed_rate": round(removed / raw_count, 4) if raw_count else 0.0,
    }


def _llm_verify_filter(
    keyphrases_per_doc: List[List[str]],
    texts: List[str],
    llm: OllamaClient,
    model: str,
) -> List[List[str]]:
    """LLM이 각 키프레이즈의 원문 근거 여부를 직접 판정 (문서당 1회, 캐시됨)."""
    filtered: List[List[str]] = []
    for doc_idx, (text, kps) in enumerate(zip(texts, keyphrases_per_doc)):
        if not kps:
            filtered.append([])
            continue
        kp_lines = "\n".join(f"{i}: {kp}" for i, kp in enumerate(kps))
        prompt = f"""다음 문서와 키프레이즈 목록이 있습니다. 각 키프레이즈가 문서 내용에 실제로 근거하는지 판정하세요.

문서:
{text[:2000]}

키프레이즈:
{kp_lines}

문서에 근거한 키프레이즈의 번호만 JSON 배열로 출력하세요. 예: [0, 2, 3]"""
        try:
            raw = llm.chat_text(
                model,
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
                cache_tag=f"halluc_verify:{text_hash(text)}:{text_hash('|'.join(kps))}",
            )
            kept_indices = {int(i) for i in parse_json_response(raw) if isinstance(i, (int, float))}
            filtered.append([kp for i, kp in enumerate(kps) if i in kept_indices])
        except Exception as e:
            logger.warning("LLM 환각 검증 실패 (문서 %d), 전체 유지: %s", doc_idx, e)
            filtered.append(list(kps))
    return filtered
