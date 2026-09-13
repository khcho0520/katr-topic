"""3단계: 클러스터링 — ablation 지점.

- two_level_kmeans: QualIT 논문의 2층 K-means (실루엣 k 선택, MCPanal 이식)
- kmeans: 단층 K-means (실루엣 k 선택)
- hdbscan: 밀도 기반 (개선 후보: 아웃라이어 허용, k 자동)

시드 의미론: random_state=seed — 시드별 반복실험에서 유일하게 변하는 단계.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClusteringOutput:
    main_labels: np.ndarray                      # 항목별 main 클러스터 (-1 = 노이즈)
    sub_labels: Optional[np.ndarray] = None      # 항목별 sub 클러스터 (2층일 때)
    sub_topics: Dict[int, Dict[int, List[int]]] = field(default_factory=dict)
    chosen_k: int = 0


def safe_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    try:
        n = len(embeddings)
        uniq = np.unique(labels[labels >= 0]) if (labels < 0).any() else np.unique(labels)
        if n < 3 or len(uniq) < 2 or len(uniq) >= n:
            return 0.0
        mask = labels >= 0
        if mask.sum() < 3 or len(np.unique(labels[mask])) < 2:
            return 0.0
        return float(silhouette_score(embeddings[mask], labels[mask]))
    except Exception:
        return 0.0


def _kmeans_best_k(
    embeddings: np.ndarray,
    min_k: int,
    max_k: int,
    seed: int,
    target_k: Optional[int] = None,
) -> np.ndarray:
    from sklearn.cluster import KMeans

    n = len(embeddings)
    if target_k is not None:
        k = max(1, min(int(target_k), n))
        return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(embeddings)

    best_k, best_score = min_k, -1.0
    for k in range(min_k, min(max_k + 1, n // 2 + 1)):
        try:
            labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            score = safe_silhouette(embeddings, labels)
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            continue
    return KMeans(n_clusters=best_k, random_state=seed, n_init=10).fit_predict(embeddings)


def two_level_kmeans(
    embeddings: np.ndarray,
    seed: int,
    min_k_main: int = 2,
    max_k_main: int = 10,
    min_k_sub: int = 2,
    max_k_sub: int = 5,
    target_k_main: Optional[int] = None,
) -> ClusteringOutput:
    """QualIT 논문의 2층 클러스터링. main cluster가 8개 이상 항목일 때만 sub 분할."""
    from sklearn.cluster import KMeans

    n = len(embeddings)
    if n < min_k_main * 2:
        k = min(2, n)
        labels = KMeans(n_clusters=max(1, k), random_state=seed, n_init=10).fit_predict(embeddings)
        return ClusteringOutput(
            main_labels=labels,
            sub_labels=np.zeros(n, dtype=int),
            chosen_k=len(set(labels)),
        )

    main_labels = _kmeans_best_k(embeddings, min_k_main, max_k_main, seed, target_k_main)

    sub_labels = np.full(n, 0, dtype=int)
    sub_topics: Dict[int, Dict[int, List[int]]] = {}
    min_items_for_sub = max(min_k_sub * 3, 8)   # 과분해 방지 (MCPanal 기준)

    for main_id in sorted(set(int(m) for m in main_labels)):
        indices = np.where(main_labels == main_id)[0]
        if len(indices) < min_items_for_sub:
            sub_topics[main_id] = {0: indices.tolist()}
            continue
        local_labels = _kmeans_best_k(embeddings[indices], min_k_sub, max_k_sub, seed)
        topics: Dict[int, List[int]] = {}
        for sub_id in sorted(set(int(s) for s in local_labels)):
            local_pos = np.where(local_labels == sub_id)[0]
            global_idx = indices[local_pos]
            topics[sub_id] = global_idx.tolist()
            sub_labels[global_idx] = sub_id
        sub_topics[main_id] = topics

    return ClusteringOutput(
        main_labels=np.asarray(main_labels, dtype=int),
        sub_labels=sub_labels,
        sub_topics=sub_topics,
        chosen_k=len(set(int(m) for m in main_labels)),
    )


def single_kmeans(
    embeddings: np.ndarray,
    seed: int,
    min_k: int = 2,
    max_k: int = 15,
    target_k: Optional[int] = None,
) -> ClusteringOutput:
    labels = _kmeans_best_k(embeddings, min_k, max_k, seed, target_k)
    return ClusteringOutput(main_labels=np.asarray(labels, dtype=int), chosen_k=len(set(labels)))


def hdbscan_clustering(
    embeddings: np.ndarray,
    seed: int,                      # HDBSCAN은 결정적이지만 계약 일관성을 위해 유지
    min_cluster_size: int = 0,
) -> ClusteringOutput:
    from hdbscan import HDBSCAN

    n = len(embeddings)
    mcs = min_cluster_size or max(3, n // 20)
    labels = HDBSCAN(
        min_cluster_size=mcs,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(embeddings)
    valid = sorted(set(int(l) for l in labels if l >= 0))
    if not valid:
        # 전부 노이즈면 K-means 폴백
        logger.warning("HDBSCAN 전체 노이즈 → K-means 폴백")
        return single_kmeans(embeddings, seed)
    return ClusteringOutput(main_labels=np.asarray(labels, dtype=int), chosen_k=len(valid))


def run_clustering(
    method: str,
    embeddings: np.ndarray,
    seed: int,
    target_k: Optional[int] = None,
    max_k: int = 10,
) -> ClusteringOutput:
    if method == "two_level_kmeans":
        return two_level_kmeans(embeddings, seed, max_k_main=max_k, target_k_main=target_k)
    if method == "kmeans":
        return single_kmeans(embeddings, seed, max_k=max(max_k, 15), target_k=target_k)
    if method == "hdbscan":
        return hdbscan_clustering(embeddings, seed)
    raise ValueError(f"알 수 없는 클러스터링 방법: {method}")
