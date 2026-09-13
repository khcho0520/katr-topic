"""BERTopic 전수할당(assign_outliers) 변형 검증."""
from __future__ import annotations

import numpy as np

from core.models.bertopic_model import BERTopicConfig, BERTopicModel
from tests.stubs import blob_embeddings, synthetic_corpus


def test_assign_outliers_gives_full_coverage():
    corpus = synthetic_corpus(15)
    embeddings, _ = blob_embeddings(15, dim=32, n_clusters=2)

    base = BERTopicModel(BERTopicConfig(min_cluster_size=4)).fit(corpus, embeddings, 42)
    full = BERTopicModel(
        BERTopicConfig(min_cluster_size=4, assign_outliers=True)
    ).fit(corpus, embeddings, 42)

    # 전수할당: 아웃라이어 없음
    assert int(np.sum(full.assignments == -1)) == 0
    # 발견 토픽 집합은 동일 (할당 확장만 수행, 새 토픽 생성 없음)
    assert set(full.topic_ids) == set(base.topic_ids)
    # 원래 할당됐던 문서의 소속은 변하지 않음
    assigned_mask = base.assignments >= 0
    assert np.array_equal(base.assignments[assigned_mask], full.assignments[assigned_mask])
    # 확장 전 아웃라이어 수가 기록됨
    assert full.artifacts["n_outliers_before_assign"] == int(np.sum(base.assignments == -1))
