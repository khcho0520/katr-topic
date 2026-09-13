"""QualIT 계열 파이프라인: 설정으로 각 단계를 교체하는 ablation 프레임워크.

QualITConfig 기본값 = QualIT 논문 충실 재현 (Kapoor et al., arXiv:2409.15626).
필드를 바꾸면 개선 변형이 된다 — 코드 분기 없이 설정만으로 탐색공간을 표현한다.

시드 의미론: 키프레이즈 추출·환각필터·라벨링은 시드 무관(캐시 재사용),
클러스터링만 seed에 따라 변한다. 논문 방법론에 명시할 것.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np

from config import DEFAULT_LLM_MODEL
from core.llm.embeddings import Embedder, get_default_embedder
from core.llm.ollama_client import OllamaClient, get_default_client
from core.models.base import ModelConfig, TopicModel, register
from core.models.qualit import hallucination, labeling, refine, representation
from core.models.qualit.candidates import select_candidate_indices
from core.models.qualit.clustering import run_clustering, safe_silhouette
from core.models.qualit.keyphrase import extract_keyphrases
from core.types import Corpus, TopicResult
from core.utils.progress import ProgressCallback, sub_progress

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualITConfig(ModelConfig):
    name: str = "qualit"
    n_topics: Optional[int] = None                 # None = 실루엣/HDBSCAN 자동
    max_topics: int = 10
    # 1단계: 키프레이즈 추출
    keyphrase_model: str = DEFAULT_LLM_MODEL
    keyphrases_per_doc: Optional[int] = None       # None = 문서 길이 기반 자동
    # 2단계: 환각 필터
    hallucination_filter: Literal["cosine", "percentile", "llm_verify", "none"] = "cosine"
    hallucination_threshold: float = 0.10
    # 3단계: 클러스터링 표현·방법
    representation: Literal["keyphrase", "doc", "joint"] = "keyphrase"
    joint_alpha: float = 0.5
    clustering: Literal["two_level_kmeans", "kmeans", "hdbscan"] = "two_level_kmeans"
    # 4단계: 키워드·라벨
    keywords: Literal["llm", "ctfidf"] = "llm"
    label_model: str = DEFAULT_LLM_MODEL
    # 5단계(선택): 유사 토픽 병합
    llm_refine: bool = False
    refine_threshold: float = 0.88
    # 문서 할당: 키프레이즈 없는 문서를 최근접 센트로이드로 할당할지
    assign_all: bool = True
    # 대량 코퍼스 보호: LLM 키프레이즈 추출 대상 대표 문서 상한 (0 = 무제한).
    # 초과 시 임베딩 층화 샘플링으로 대표 문서만 추출하고, 나머지는 assign_all로 할당.
    max_llm_docs: int = 500


@register("qualit", QualITConfig)
class QualITModel(TopicModel):
    requires_embeddings = True
    requires_llm = True

    def __init__(
        self,
        config: ModelConfig,
        llm: Optional[OllamaClient] = None,
        embedder: Optional[Embedder] = None,
    ):
        super().__init__(config)
        self._llm = llm
        self._embedder = embedder

    @property
    def llm(self) -> OllamaClient:
        if self._llm is None:
            self._llm = get_default_client()
        return self._llm

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_default_embedder()
        return self._embedder

    def fit(
        self,
        corpus: Corpus,
        embeddings: Optional[np.ndarray],
        seed: int,
        progress: Optional[ProgressCallback] = None,
    ) -> TopicResult:
        if embeddings is None:
            raise ValueError("QualIT은 공유 문서 임베딩이 필요합니다.")
        cfg: QualITConfig = self.config  # type: ignore[assignment]
        started = time.perf_counter()
        texts = corpus.texts
        doc_embeddings = np.asarray(embeddings, dtype=np.float32)
        artifacts: Dict[str, object] = {}

        # 0. 대량 코퍼스면 대표 문서만 LLM 대상 선정 (결정적 → 시드 간 캐시 재사용)
        candidate_indices = select_candidate_indices(doc_embeddings, cfg.max_llm_docs)
        is_sampled = len(candidate_indices) < len(texts)
        artifacts["n_llm_docs"] = len(candidate_indices)
        if is_sampled and progress:
            progress(0.01, f"대표 문서 선택: {len(texts):,}건 → {len(candidate_indices):,}건")

        # 1. LLM 키프레이즈 추출 (시드 무관, 캐시됨)
        if progress:
            progress(0.02, "키프레이즈 추출 시작")
        keyphrases_per_doc = extract_keyphrases(
            texts,
            self.llm,
            cfg.keyphrase_model,
            max_per_doc=cfg.keyphrases_per_doc,
            candidate_indices=candidate_indices if is_sampled else None,
            progress=sub_progress(progress, 0.02, 0.45),
        )

        # 키프레이즈 임베딩 (고유값만, 캐시됨)
        unique_kps = sorted({kp for kps in keyphrases_per_doc for kp in kps})
        if not unique_kps and cfg.representation != "doc":
            raise RuntimeError("추출된 키프레이즈가 없습니다 (LLM 응답 확인).")
        kp_matrix = (
            self.embedder.embed_texts(unique_kps, progress=sub_progress(progress, 0.45, 0.55))
            if unique_kps
            else np.zeros((0, doc_embeddings.shape[1]), dtype=np.float32)
        )
        kp_lookup = {kp: kp_matrix[i] for i, kp in enumerate(unique_kps)}

        # 2. 환각 필터 (시드 무관)
        if progress:
            progress(0.56, f"환각 필터 ({cfg.hallucination_filter})")
        filtered_kps, halluc_stats = hallucination.filter_keyphrases(
            cfg.hallucination_filter,
            keyphrases_per_doc,
            doc_embeddings,
            kp_lookup,
            threshold=cfg.hallucination_threshold,
            llm=self.llm,
            llm_model=cfg.keyphrase_model,
            texts=texts,
        )
        artifacts.update(halluc_stats)

        # 3. 표현 구성 + 클러스터링 (시드 의존)
        rep = representation.build_representation(
            cfg.representation, filtered_kps, doc_embeddings, kp_lookup, cfg.joint_alpha
        )
        if progress:
            progress(0.62, f"클러스터링 ({cfg.clustering}, items={len(rep.vectors)})")
        cluster_out = run_clustering(
            cfg.clustering, rep.vectors, seed, target_k=cfg.n_topics, max_k=cfg.max_topics
        )
        main_labels = cluster_out.main_labels

        # 4. (선택) 유사 토픽 병합
        if cfg.llm_refine:
            main_labels, _ = refine.merge_similar_topics(rep.vectors, main_labels, cfg.refine_threshold)

        # 5. 문서 할당
        assignments = self._assign_documents(
            cfg, corpus, doc_embeddings, rep, main_labels
        )

        # 6. 키워드·라벨
        if progress:
            progress(0.8, "토픽 키워드·라벨 생성")
        topic_ids = sorted(t for t in set(int(a) for a in assignments) if t >= 0)
        keywords = self._build_keywords(cfg, corpus, rep, main_labels, assignments, topic_ids)
        labels = self._build_labels(cfg, corpus, rep, main_labels, assignments, topic_ids, keywords)

        artifacts.update(
            {
                "n_keyphrases_clustered": len(rep.vectors),
                "chosen_k": cluster_out.chosen_k,
                "keyphrase_silhouette": float(safe_silhouette(rep.vectors, main_labels)),
                "keyphrases_per_doc": filtered_kps,
            }
        )
        if progress:
            progress(1.0, "QualIT 완료")
        return TopicResult(
            model_name=cfg.label,
            config_hash=cfg.hash(),
            seed=seed,
            assignments=assignments,
            topic_keywords=keywords,
            topic_labels=labels,
            artifacts=artifacts,
            runtime_sec=time.perf_counter() - started,
        )

    # ---- 내부 단계 ----

    def _assign_documents(
        self,
        cfg: QualITConfig,
        corpus: Corpus,
        doc_embeddings: np.ndarray,
        rep: representation.RepresentationOutput,
        main_labels: np.ndarray,
    ) -> np.ndarray:
        n_docs = len(corpus)
        if cfg.representation == "doc":
            return np.asarray(main_labels, dtype=int)

        # 문서별 키프레이즈 다수결
        assignments = np.full(n_docs, -1, dtype=int)
        votes: Dict[int, Dict[int, int]] = {}
        for item_idx, doc_idx in enumerate(rep.item_doc_idx):
            label = int(main_labels[item_idx])
            if label < 0:
                continue
            votes.setdefault(doc_idx, {})
            votes[doc_idx][label] = votes[doc_idx].get(label, 0) + 1
        for doc_idx, counter in votes.items():
            assignments[doc_idx] = max(counter.items(), key=lambda kv: (kv[1], -kv[0]))[0]

        # 키프레이즈가 없거나 전부 노이즈인 문서 → 최근접 센트로이드 (선택)
        if cfg.assign_all:
            topic_ids = sorted(set(int(l) for l in main_labels if l >= 0))
            if topic_ids:
                centroids = np.stack(
                    [rep.vectors[main_labels == tid].mean(axis=0) for tid in topic_ids]
                )
                c_norm = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
                unassigned = np.where(assignments < 0)[0]
                if len(unassigned):
                    d = doc_embeddings[unassigned]
                    d_norm = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
                    sims = d_norm @ c_norm.T
                    nearest = np.argmax(sims, axis=1)
                    for pos, doc_idx in enumerate(unassigned):
                        assignments[doc_idx] = topic_ids[int(nearest[pos])]
        return assignments

    def _build_keywords(
        self,
        cfg: QualITConfig,
        corpus: Corpus,
        rep: representation.RepresentationOutput,
        main_labels: np.ndarray,
        assignments: np.ndarray,
        topic_ids: List[int],
    ) -> Dict[int, List[str]]:
        if cfg.keywords == "ctfidf":
            if corpus.tokenized is None:
                raise ValueError("ctfidf 키워드에는 corpus.tokenized가 필요합니다.")
            topic_token_docs = {
                tid: [corpus.tokenized[i] for i in np.where(assignments == tid)[0]]
                for tid in topic_ids
            }
            return labeling.ctfidf_keywords(topic_token_docs)

        # "llm": 멤버 키프레이즈 빈도 상위 (QualIT 논문)
        keywords: Dict[int, List[str]] = {}
        for tid in topic_ids:
            member_kps = [
                rep.item_keyphrases[i]
                for i in np.where(main_labels == tid)[0]
                if rep.item_keyphrases[i]
            ]
            keywords[tid] = labeling.keyphrase_keywords(member_kps)
        return keywords

    def _build_labels(
        self,
        cfg: QualITConfig,
        corpus: Corpus,
        rep: representation.RepresentationOutput,
        main_labels: np.ndarray,
        assignments: np.ndarray,
        topic_ids: List[int],
        keywords: Dict[int, List[str]],
    ) -> Dict[int, str]:
        texts = corpus.texts
        labels: Dict[int, str] = {}
        accumulated: List[str] = []
        # 큰 토픽부터 라벨링 (contrastive 문맥 안정화)
        sizes = {tid: int(np.sum(assignments == tid)) for tid in topic_ids}
        for tid in sorted(topic_ids, key=lambda t: -sizes[t]):
            kws = keywords.get(tid) or []
            sample_texts = [texts[i] for i in np.where(assignments == tid)[0][:2]]
            theme = labeling.distill_theme(
                kws, sample_texts, self.llm, cfg.label_model, other_themes=accumulated
            )
            labels[tid] = theme or ", ".join(kws[:4])
            accumulated.append(labels[tid])
        return labels
