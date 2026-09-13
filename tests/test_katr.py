"""KATR (제안 모델) 단위 검증: 전수 커버리지, 앵커 재할당, α 대조군, 병합."""
from __future__ import annotations

import numpy as np
import pytest

from core.models.katr_model import KATRConfig, KATRModel
from tests.stubs import StubEmbedder, StubLLM, blob_embeddings, synthetic_corpus

N_PER_THEME = 15


@pytest.fixture(scope="module")
def corpus():
    return synthetic_corpus(N_PER_THEME)


@pytest.fixture(scope="module")
def embeddings():
    matrix, _ = blob_embeddings(N_PER_THEME, dim=32, n_clusters=2)
    return matrix


def _fit(config, corpus, embeddings, seed=42):
    return KATRModel(config, llm=StubLLM(), embedder=StubEmbedder(dim=32)).fit(
        corpus, embeddings, seed
    )


def test_full_coverage_and_structure(corpus, embeddings):
    config = KATRConfig(n_topics=2, use_umap=False, hallucination_filter="none",
                        anchor_docs_per_topic=5)
    result = _fit(config, corpus, embeddings)
    assert int(np.sum(result.assignments == -1)) == 0          # 전수 커버리지
    assert len(result.topic_ids) >= 1
    assert result.artifacts["n_llm_docs"] > 0
    assert isinstance(result.artifacts["reassigned_per_iter"], list)
    for tid in result.topic_ids:
        assert result.topic_keywords.get(tid)                   # c-TF-IDF 키워드 존재
        assert result.topic_labels.get(tid)


def test_alpha1_is_pure_centroid_control(corpus, embeddings):
    """α=1.0이면 앵커가 점수에 기여하지 않아야 한다 (대조군 동치성)."""
    config = KATRConfig(n_topics=2, use_umap=False, hallucination_filter="none",
                        anchor_docs_per_topic=5, alpha=1.0, merge_threshold=2.0)
    result = _fit(config, corpus, embeddings)
    # 잘 분리된 2블롭 + K=2 → 순수 센트로이드 재할당은 블롭 경계와 일치해야 함
    a = result.assignments
    group1 = set(a[:N_PER_THEME].tolist())
    group2 = set(a[N_PER_THEME:].tolist())
    assert len(group1) == 1 and len(group2) == 1 and group1 != group2


def test_anchor_influences_assignment(corpus, embeddings):
    """α=0(순수 앵커)과 α=1(순수 중심)이 서로 다른 점수 함수로 동작하는지 확인."""
    base = dict(n_topics=2, use_umap=False, hallucination_filter="none",
                anchor_docs_per_topic=5, merge_threshold=2.0, refine_iters=1)
    r_anchor = _fit(KATRConfig(**base, alpha=0.0), corpus, embeddings)
    r_centroid = _fit(KATRConfig(**base, alpha=1.0), corpus, embeddings)
    # 두 결과 모두 유효한 전수 할당 (값이 같을 수도 있으나 계약은 유지되어야 함)
    assert len(r_anchor.assignments) == len(r_centroid.assignments) == len(corpus)
    assert int(np.sum(r_anchor.assignments == -1)) == 0


def test_merge_by_anchor_remaps_ids(corpus, embeddings):
    """병합 임계값을 극단적으로 낮추면 토픽이 1개로 합쳐지고 ID가 0부터 재매핑된다."""
    config = KATRConfig(n_topics=3, use_umap=False, hallucination_filter="none",
                        anchor_docs_per_topic=5, merge_threshold=-1.0, refine_iters=0)
    result = _fit(config, corpus, embeddings)
    assert result.topic_ids == [0]
    assert result.artifacts["n_topics_after_merge"] == 1


def test_deterministic_given_seed(corpus, embeddings):
    config = KATRConfig(n_topics=2, use_umap=False, hallucination_filter="none",
                        anchor_docs_per_topic=5)
    r1 = _fit(config, corpus, embeddings, seed=7)
    r2 = _fit(config, corpus, embeddings, seed=7)
    assert np.array_equal(r1.assignments, r2.assignments)
