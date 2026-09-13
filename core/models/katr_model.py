"""KATR (Keyphrase-Anchored Topic Refinement) — 제안 모델.

핵심 아이디어: LLM을 QualIT처럼 "표현 생성기"로 쓰지 않고 "클러스터 정제 신호(앵커)"로
재배치한다. BERTopic의 구조적 강점(UMAP 다양체 + 문서 직접 클러스터링 + c-TF-IDF)을
채택하고, 그 위에 LLM 키프레이즈 앵커로 토픽 경계를 재할당·정제한다.

파이프라인:
1. 초기 클러스터링 — 문서 임베딩 → (선택) UMAP 축소 → K-means(K) 또는 HDBSCAN+최근접 할당
2. 토픽 앵커 — 코퍼스 수준 고정 후보 풀(결정적, 시드 무관 → LLM 캐시가 시드·모델 간 공유)
   에서 토픽별 대표 문서를 골라 LLM 키프레이즈 추출 → 환각 필터 → 앵커 = 키프레이즈 임베딩 평균
3. 앵커 유도 재할당 — score(d,t) = α·cos(e_d, centroid_t) + (1−α)·cos(e_d, anchor_t)
   argmax로 재할당, 수렴까지 반복. α=1.0이면 앵커 무효(순수 기하 중심) 대조군.
4. 키워드 = c-TF-IDF(Kiwi 토큰), 라벨 = 앵커 키프레이즈의 contrastive LLM 증류,
   (선택) 앵커 유사도 기반 중복 토픽 병합.

시드 의미론: UMAP/K-means 초기화만 시드 의존. 후보 풀·키프레이즈·라벨은 캐시 재사용.
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
from core.models.qualit import hallucination, labeling
from core.models.qualit.candidates import select_candidate_indices
from core.models.qualit.keyphrase import extract_keyphrases
from core.types import Corpus, TopicResult
from core.utils.progress import ProgressCallback, sub_progress

logger = logging.getLogger(__name__)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


@dataclass(frozen=True)
class KATRConfig(ModelConfig):
    name: str = "katr"
    n_topics: Optional[int] = None          # None = HDBSCAN 자동, int = K-means K
    # 1단계: 초기 클러스터링
    use_umap: bool = True
    umap_n_components: int = 5
    umap_n_neighbors: int = 15
    min_cluster_size: int = 0               # HDBSCAN용 (0 = max(3, n/20))
    # 2단계: 앵커 구축
    anchor_pool_size: int = 1000            # 코퍼스 수준 고정 LLM 후보 풀 (시드 무관 캐시)
    anchor_docs_per_topic: int = 15
    keyphrase_model: str = DEFAULT_LLM_MODEL
    keyphrases_per_doc: Optional[int] = None  # None = 문서 길이 기반 자동 (QualIT과 캐시 공유)
    hallucination_filter: Literal["cosine", "percentile", "none"] = "percentile"
    hallucination_threshold: float = 0.10
    # 3단계: 앵커 유도 재할당
    alpha: float = 0.6                      # 기하 중심 비중 (1.0 = 앵커 무효 대조군)
    refine_iters: int = 3
    # 4단계: 키워드·라벨·병합
    label_model: str = DEFAULT_LLM_MODEL
    llm_label: bool = True
    merge_threshold: float = 0.97           # 앵커 코사인 유사도 병합 (>1 = 병합 없음)
    # c-TF-IDF 키워드를 센트로이드 최근접 상위 비율의 "코어 문서"로만 계산.
    # BERTopic이 HDBSCAN 밀도 코어에서 키워드를 뽑아 NPMI가 높은 것에 대응
    # (경계 문서의 잡음 토큰이 판별 키워드를 희석하는 것을 방지). 1.0 = 전체 사용.
    keyword_core_fraction: float = 1.0


@register("katr", KATRConfig)
class KATRModel(TopicModel):
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

    # ------------------------------------------------------------------
    def fit(
        self,
        corpus: Corpus,
        embeddings: Optional[np.ndarray],
        seed: int,
        progress: Optional[ProgressCallback] = None,
    ) -> TopicResult:
        if embeddings is None:
            raise ValueError("KATR은 공유 문서 임베딩이 필요합니다.")
        cfg: KATRConfig = self.config  # type: ignore[assignment]
        started = time.perf_counter()
        emb = np.asarray(embeddings, dtype=np.float32)
        emb_norm = _normalize(emb)
        artifacts: Dict[str, object] = {"alpha": cfg.alpha}

        # ---- 1. 초기 클러스터링 (시드 의존) ----
        if progress:
            progress(0.02, "초기 클러스터링 (UMAP + 문서 임베딩)")
        assignments = self._initial_clustering(emb, seed, cfg)
        artifacts["n_topics_initial"] = len(set(int(a) for a in assignments if a >= 0))

        # ---- 2. 토픽 앵커 구축 (시드 무관 캐시 활용) ----
        if progress:
            progress(0.15, "토픽 앵커 구축 (LLM 키프레이즈)")
        anchors, anchor_keyphrases, n_llm_docs = self._build_anchors(
            corpus, emb, emb_norm, assignments, cfg,
            progress=sub_progress(progress, 0.15, 0.7),
        )
        artifacts["n_llm_docs"] = n_llm_docs

        # ---- 3. 앵커 유도 재할당 ----
        if progress:
            progress(0.72, f"앵커 유도 재할당 (α={cfg.alpha}, {cfg.refine_iters}회)")
        assignments, reassign_log = self._anchor_refinement(
            emb_norm, assignments, anchors, cfg
        )
        artifacts["reassigned_per_iter"] = reassign_log

        # ---- 4. (선택) 앵커 유사도 병합 ----
        if cfg.merge_threshold <= 1.0:
            assignments, old_to_new = self._merge_by_anchor(assignments, anchors, cfg.merge_threshold)
            if old_to_new:
                merged_anchor_kps: Dict[int, List[str]] = {}
                for old_id, new_id in old_to_new.items():
                    merged_anchor_kps.setdefault(new_id, []).extend(anchor_keyphrases.get(old_id, []))
                anchor_keyphrases = {
                    tid: labeling.keyphrase_keywords(kps, top_k=15)
                    for tid, kps in merged_anchor_kps.items()
                }
            artifacts["n_topics_after_merge"] = int(len(set(int(a) for a in assignments if a >= 0)))

        topic_ids = sorted(set(int(a) for a in assignments if a >= 0))

        # ---- 5. 키워드(c-TF-IDF) + 라벨(LLM contrastive) ----
        if progress:
            progress(0.85, "키워드·라벨 생성")
        if corpus.tokenized is None:
            raise ValueError("KATR c-TF-IDF에는 corpus.tokenized가 필요합니다.")
        topic_token_docs = {}
        for tid in topic_ids:
            member = np.where(assignments == tid)[0]
            if cfg.keyword_core_fraction < 1.0 and len(member) > 3:
                centroid = _normalize(emb[member].mean(axis=0, keepdims=True))
                sims = (emb_norm[member] @ centroid.T).ravel()
                n_core = max(3, int(np.ceil(len(member) * cfg.keyword_core_fraction)))
                member = member[np.argsort(-sims)[:n_core]]
            topic_token_docs[tid] = [corpus.tokenized[i] for i in member]
        keywords = labeling.ctfidf_keywords(topic_token_docs)
        labels = self._build_labels(corpus, assignments, topic_ids, keywords, anchor_keyphrases, cfg)

        artifacts["anchor_keyphrases"] = {
            str(t): kps[:10] for t, kps in anchor_keyphrases.items()
        }
        if progress:
            progress(1.0, "KATR 완료")
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

    # ------------------------------------------------------------------
    def _initial_clustering(self, emb: np.ndarray, seed: int, cfg: KATRConfig) -> np.ndarray:
        n = len(emb)
        if cfg.use_umap and n > 20:
            from umap import UMAP

            reducer = UMAP(
                n_neighbors=min(cfg.umap_n_neighbors, max(2, n - 1)),
                n_components=min(cfg.umap_n_components, max(2, n - 2)),
                min_dist=0.0,
                metric="cosine",
                random_state=seed,
            )
            space = np.asarray(reducer.fit_transform(emb), dtype=np.float32)
        else:
            space = emb

        if cfg.n_topics:
            from sklearn.cluster import KMeans

            k = max(2, min(int(cfg.n_topics), n - 1))
            return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(space).astype(int)

        # K 미지정: HDBSCAN + 최근접 센트로이드 전수 할당 (BERTopic-full과 동일 규칙)
        from hdbscan import HDBSCAN

        mcs = cfg.min_cluster_size or max(3, n // 20)
        labels = HDBSCAN(min_cluster_size=mcs, metric="euclidean",
                         cluster_selection_method="eom").fit_predict(space).astype(int)
        topic_ids = sorted(t for t in set(labels) if t >= 0)
        if not topic_ids:
            from sklearn.cluster import KMeans

            return KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(space).astype(int)
        centroids = _normalize(np.stack([space[labels == t].mean(axis=0) for t in topic_ids]))
        outliers = np.where(labels == -1)[0]
        if len(outliers):
            s_norm = _normalize(space[outliers])
            nearest = np.argmax(s_norm @ centroids.T, axis=1)
            for pos, doc_idx in enumerate(outliers):
                labels[doc_idx] = topic_ids[int(nearest[pos])]
        return labels

    # ------------------------------------------------------------------
    def _build_anchors(
        self,
        corpus: Corpus,
        emb: np.ndarray,
        emb_norm: np.ndarray,
        assignments: np.ndarray,
        cfg: KATRConfig,
        progress: Optional[ProgressCallback] = None,
    ):
        """토픽별 앵커 벡터. LLM 대상은 코퍼스 수준 고정 풀에서만 골라 캐시를 공유한다."""
        texts = corpus.texts
        pool = select_candidate_indices(emb, cfg.anchor_pool_size)
        pool_set = set(pool)

        topic_ids = sorted(set(int(a) for a in assignments if a >= 0))
        # 토픽별 대표: 풀 소속 멤버 중 토픽 센트로이드에 가까운 순 상위 m건
        anchor_doc_indices: List[int] = []
        docs_by_topic: Dict[int, List[int]] = {}
        for tid in topic_ids:
            member = np.where(assignments == tid)[0]
            member_in_pool = [i for i in member if i in pool_set]
            if not member_in_pool:
                member_in_pool = member.tolist()[: cfg.anchor_docs_per_topic]
            centroid = _normalize(emb[member].mean(axis=0, keepdims=True))
            sims = (emb_norm[member_in_pool] @ centroid.T).ravel()
            ranked = [member_in_pool[i] for i in np.argsort(-sims)]
            chosen = ranked[: cfg.anchor_docs_per_topic]
            docs_by_topic[tid] = chosen
            anchor_doc_indices.extend(chosen)

        anchor_doc_indices = sorted(set(anchor_doc_indices))
        keyphrases_all = extract_keyphrases(
            texts,
            self.llm,
            cfg.keyphrase_model,
            max_per_doc=cfg.keyphrases_per_doc,
            candidate_indices=anchor_doc_indices,
            progress=progress,
        )

        # 환각 필터 (원공간 코사인)
        unique_kps = sorted({kp for kps in keyphrases_all for kp in kps})
        kp_matrix = (
            self.embedder.embed_texts(unique_kps)
            if unique_kps
            else np.zeros((0, emb.shape[1]), dtype=np.float32)
        )
        kp_lookup = {kp: kp_matrix[i] for i, kp in enumerate(unique_kps)}
        if cfg.hallucination_filter != "none":
            keyphrases_all, _stats = hallucination.filter_keyphrases(
                cfg.hallucination_filter, keyphrases_all, emb, kp_lookup,
                threshold=cfg.hallucination_threshold,
            )

        anchors: Dict[int, np.ndarray] = {}
        anchor_keyphrases: Dict[int, List[str]] = {}
        for tid in topic_ids:
            kps: List[str] = []
            vectors: List[np.ndarray] = []
            for doc_idx in docs_by_topic[tid]:
                for kp in keyphrases_all[doc_idx]:
                    vec = kp_lookup.get(kp)
                    if vec is not None:
                        kps.append(kp)
                        vectors.append(vec)
            if vectors:
                anchor = np.mean(np.stack(vectors), axis=0)
            else:
                # 폴백: 키프레이즈가 전무하면 기하 센트로이드 (앵커 = 중심)
                member = np.where(assignments == tid)[0]
                anchor = emb[member].mean(axis=0)
            anchors[tid] = anchor / max(float(np.linalg.norm(anchor)), 1e-9)
            anchor_keyphrases[tid] = labeling.keyphrase_keywords(kps, top_k=15)
        return anchors, anchor_keyphrases, len(anchor_doc_indices)

    # ------------------------------------------------------------------
    def _anchor_refinement(
        self,
        emb_norm: np.ndarray,
        assignments: np.ndarray,
        anchors: Dict[int, np.ndarray],
        cfg: KATRConfig,
    ):
        """score = α·cos(문서, 센트로이드) + (1−α)·cos(문서, 앵커) argmax 재할당 반복."""
        assignments = assignments.copy()
        topic_ids = sorted(anchors.keys())
        anchor_matrix = np.stack([anchors[t] for t in topic_ids])
        reassign_log: List[int] = []

        for _ in range(max(0, cfg.refine_iters)):
            centroids = []
            for tid in topic_ids:
                member = np.where(assignments == tid)[0]
                if len(member) == 0:
                    centroids.append(anchors[tid])   # 빈 토픽은 앵커로 대체
                else:
                    c = emb_norm[member].mean(axis=0)
                    centroids.append(c / max(float(np.linalg.norm(c)), 1e-9))
            centroid_matrix = np.stack(centroids)

            scores = cfg.alpha * (emb_norm @ centroid_matrix.T) \
                + (1.0 - cfg.alpha) * (emb_norm @ anchor_matrix.T)
            new_assignments = np.array([topic_ids[i] for i in np.argmax(scores, axis=1)], dtype=int)
            n_changed = int(np.sum(new_assignments != assignments))
            reassign_log.append(n_changed)
            assignments = new_assignments
            if n_changed == 0:
                break
        return assignments, reassign_log

    # ------------------------------------------------------------------
    def _merge_by_anchor(self, assignments: np.ndarray, anchors: Dict[int, np.ndarray], threshold: float):
        """앵커 코사인 ≥ threshold인 토픽 병합. (재매핑된 할당, old_id→new_id 맵) 반환.

        병합이 없으면 맵은 빈 dict (호출측에서 앵커 키프레이즈 재매핑 생략).
        """
        topic_ids = sorted(anchors.keys())
        n = len(topic_ids)
        if n < 2:
            return assignments, {}
        matrix = np.stack([anchors[t] for t in topic_ids])
        sims = matrix @ matrix.T
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if sims[i, j] >= threshold:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        roots = sorted({find(i) for i in range(n)})
        if len(roots) == n:
            return assignments, {}
        root_to_new = {r: k for k, r in enumerate(roots)}
        old_to_new = {topic_ids[i]: root_to_new[find(i)] for i in range(n)}
        merged = np.array([old_to_new.get(int(a), -1) if a >= 0 else -1 for a in assignments], dtype=int)
        logger.info("KATR: 앵커 유사도 병합 %d → %d개", n, len(roots))
        return merged, old_to_new

    # ------------------------------------------------------------------
    def _build_labels(self, corpus, assignments, topic_ids, keywords, anchor_keyphrases, cfg: KATRConfig):
        labels: Dict[int, str] = {}
        if not cfg.llm_label:
            return {tid: ", ".join((keywords.get(tid) or [])[:4]) for tid in topic_ids}
        texts = corpus.texts
        sizes = {tid: int(np.sum(assignments == tid)) for tid in topic_ids}
        accumulated: List[str] = []
        for tid in sorted(topic_ids, key=lambda t: -sizes[t]):
            # 라벨 재료: 앵커 키프레이즈 우선, 부족하면 c-TF-IDF 키워드 보충
            material = (anchor_keyphrases.get(tid) or [])[:8] + (keywords.get(tid) or [])[:4]
            sample_texts = [texts[i] for i in np.where(assignments == tid)[0][:2]]
            theme = labeling.distill_theme(
                material, sample_texts, self.llm, cfg.label_model, other_themes=accumulated
            )
            labels[tid] = theme or ", ".join((keywords.get(tid) or [])[:4])
            accumulated.append(labels[tid])
        return labels
