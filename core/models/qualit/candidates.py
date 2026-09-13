"""대량 코퍼스용 키프레이즈 추출 대표 문서 선택 (MCPanal 후보 선택 이식).

수만 건 헤드라인에 전부 LLM을 호출하는 대신, 임베딩 공간에서 층화(strata) 후
각 층에서 중심에 가까운 문서를 비례 배분으로 뽑는다.

의도적으로 **결정적**(실험 시드와 무관, random_state 고정)이다 — 후보가 시드마다
바뀌면 키프레이즈 LLM 캐시가 시드 간 재사용되지 않기 때문. 논문 방법론에는
"대표 문서 선택은 고정된 전처리 단계"로 기술한다.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# 후보 선택 자체의 고정 시드 (실험 시드와 별개, 캐시 재사용을 위해 불변)
_SELECTION_RANDOM_STATE = 0


def select_candidate_indices(
    doc_embeddings: np.ndarray,
    max_docs: int,
    n_strata: int = 0,
) -> List[int]:
    """층화 대표 선택: KMeans 층 → 층 크기 비례 배분 → 층 중심 최근접 문서.

    max_docs <= 0 이거나 코퍼스가 그보다 작으면 전체 인덱스를 반환한다.
    """
    n_docs = len(doc_embeddings)
    if max_docs <= 0 or n_docs <= max_docs:
        return list(range(n_docs))

    from sklearn.cluster import KMeans

    strata = n_strata or min(50, max(10, max_docs // 10))
    strata = min(strata, max_docs, n_docs)
    labels = KMeans(
        n_clusters=strata, random_state=_SELECTION_RANDOM_STATE, n_init=4
    ).fit_predict(doc_embeddings)

    # 층 크기 비례 배분 (최소 1건 보장)
    stratum_ids, counts = np.unique(labels, return_counts=True)
    quota = np.maximum(1, np.floor(counts / n_docs * max_docs).astype(int))
    # 배분 합이 max_docs를 넘으면 큰 층부터 깎고, 모자라면 큰 층부터 채운다
    while quota.sum() > max_docs:
        quota[int(np.argmax(quota))] -= 1
    order = np.argsort(-counts)
    i = 0
    while quota.sum() < max_docs:
        idx = int(order[i % len(order)])
        if quota[idx] < counts[idx]:
            quota[idx] += 1
        i += 1

    selected: List[int] = []
    for sid, k in zip(stratum_ids, quota):
        member = np.where(labels == sid)[0]
        centroid = doc_embeddings[member].mean(axis=0)
        distance = np.linalg.norm(doc_embeddings[member] - centroid, axis=1)
        nearest = member[np.argsort(distance)[: int(k)]]
        selected.extend(int(m) for m in nearest)

    selected = sorted(set(selected))[:max_docs]
    logger.info(
        "QualIT 대표 문서 선택: %d건 → %d건 (층 %d개)", n_docs, len(selected), strata
    )
    return selected
