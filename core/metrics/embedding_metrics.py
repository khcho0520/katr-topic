"""임베딩 기반 지표 (MCPanal 이식): coherence, separation, silhouette.

모든 모델에 대해 동일한 공유 bge-m3 행렬과 assignments만 사용한다.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def topic_coherence(embeddings: np.ndarray) -> float:
    """토픽 내 일관성: 센트로이드와 멤버 임베딩 간 평균 코사인 유사도."""
    if len(embeddings) < 2:
        return 1.0
    centroid = embeddings.mean(axis=0)
    return float(np.mean([_cosine(centroid, e) for e in embeddings]))


def topic_separation(centroids: List[np.ndarray]) -> float:
    """토픽 간 분리도: 센트로이드 쌍별 평균 코사인 거리(1-sim)."""
    if len(centroids) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            total += 1.0 - _cosine(centroids[i], centroids[j])
            count += 1
    return total / count if count else 0.0


def silhouette_assigned(embeddings: np.ndarray, assignments: np.ndarray) -> float:
    """할당된 문서만(-1 제외) 대상 실루엣."""
    from sklearn.metrics import silhouette_score

    mask = assignments >= 0
    if mask.sum() < 3:
        return 0.0
    labels = assignments[mask]
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= mask.sum():
        return 0.0
    try:
        return float(silhouette_score(embeddings[mask], labels))
    except Exception:
        return 0.0


def silhouette_with_noise(embeddings: np.ndarray, assignments: np.ndarray) -> float:
    """아웃라이어(-1)를 하나의 노이즈 토픽으로 취급한 실루엣 (커버리지 페널티 반영)."""
    from sklearn.metrics import silhouette_score

    if len(embeddings) < 3:
        return 0.0
    labels = assignments.copy()
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return 0.0
    try:
        return float(silhouette_score(embeddings, labels))
    except Exception:
        return 0.0


def compute_embedding_metrics(
    embeddings: np.ndarray, assignments: np.ndarray
) -> Dict[str, float]:
    topic_ids = sorted(t for t in set(int(a) for a in assignments) if t >= 0)
    coherences: Dict[int, float] = {}
    centroids: List[np.ndarray] = []
    for tid in topic_ids:
        member = embeddings[assignments == tid]
        if len(member) == 0:
            continue
        coherences[tid] = topic_coherence(member)
        centroids.append(member.mean(axis=0))

    return {
        "coherence_emb": float(np.mean(list(coherences.values()))) if coherences else 0.0,
        "separation": topic_separation(centroids),
        "silhouette": silhouette_assigned(embeddings, assignments),
        "silhouette_noise": silhouette_with_noise(embeddings, assignments),
    }


def per_topic_coherence(embeddings: np.ndarray, assignments: np.ndarray) -> Dict[int, float]:
    out = {}
    for tid in sorted(t for t in set(int(a) for a in assignments) if t >= 0):
        member = embeddings[assignments == tid]
        if len(member):
            out[tid] = topic_coherence(member)
    return out
