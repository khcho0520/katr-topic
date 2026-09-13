"""5단계(선택): 토픽 정제 — 유사 토픽 병합 (개선 후보, MCPanal 이식).

센트로이드 코사인 유사도가 임계값 이상인 main 토픽 쌍을 Union-Find로 병합해
질적 중복을 완화한다.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def merge_similar_topics(
    vectors: np.ndarray,
    main_labels: np.ndarray,
    similarity_threshold: float = 0.88,
) -> Tuple[np.ndarray, int]:
    """유사 토픽 병합 후 (재매핑된 라벨, 병합 후 토픽 수)를 반환. 노이즈(-1)는 유지."""
    topic_ids = sorted(set(int(l) for l in main_labels if l >= 0))
    if len(topic_ids) < 2:
        return main_labels, len(topic_ids)

    centroids = []
    for tid in topic_ids:
        member = vectors[main_labels == tid]
        centroids.append(member.mean(axis=0) if len(member) else np.zeros(vectors.shape[1]))
    centroids = np.array(centroids)

    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sim_matrix = (centroids / norms) @ (centroids / norms).T

    n = len(topic_ids)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= similarity_threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    roots = sorted({find(i) for i in range(n)})
    root_to_new = {r: new_id for new_id, r in enumerate(roots)}
    old_to_new = {topic_ids[i]: root_to_new[find(i)] for i in range(n)}

    new_labels = np.array([old_to_new.get(int(l), -1) if l >= 0 else -1 for l in main_labels], dtype=int)
    n_after = len(roots)
    if n_after < len(topic_ids):
        logger.info("유사 토픽 병합: %d개 → %d개 (임계값 %.2f)", len(topic_ids), n_after, similarity_threshold)
    return new_labels, n_after
