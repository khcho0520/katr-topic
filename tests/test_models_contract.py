"""모든 등록 모델이 TopicModel 계약을 지키는지 검증 (LLM/임베딩 스텁 사용)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from core.models import MODEL_REGISTRY, PRESETS, create_model
from core.models.katr_model import KATRConfig, KATRModel
from core.models.qualit.pipeline import QualITConfig, QualITModel
from core.types import TopicResult
from tests.stubs import StubEmbedder, StubLLM, blob_embeddings, synthetic_corpus

N_PER_THEME = 15


@pytest.fixture(scope="module")
def corpus():
    return synthetic_corpus(N_PER_THEME)


@pytest.fixture(scope="module")
def embeddings():
    matrix, _ = blob_embeddings(N_PER_THEME, dim=32, n_clusters=2)
    return matrix


def _make_model(name: str, config):
    if name == "qualit":
        return QualITModel(config, llm=StubLLM(), embedder=StubEmbedder(dim=32))
    if name == "katr":
        return KATRModel(config, llm=StubLLM(), embedder=StubEmbedder(dim=32))
    return create_model(config)


def _contract_config(name: str, config_cls):
    if name == "qualit":
        return QualITConfig(
            n_topics=2,
            hallucination_filter="none",   # 스텁 임베딩의 코사인 분포에 의존하지 않도록
            clustering="kmeans",
            keywords="llm",
        )
    if name == "katr":
        return KATRConfig(
            n_topics=2,
            use_umap=False,                # 소표본에서 UMAP 불안정 회피
            hallucination_filter="none",
            anchor_docs_per_topic=5,
        )
    config = config_cls()
    if config.n_topics is not None:
        config = dataclasses.replace(config, n_topics=3)
    return config


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY.keys()))
def test_model_contract(name, corpus, embeddings):
    _, config_cls = MODEL_REGISTRY[name]
    config = _contract_config(name, config_cls)
    model = _make_model(name, config)
    result = model.fit(corpus, embeddings, seed=42)

    assert isinstance(result, TopicResult)
    assert len(result.assignments) == len(corpus)
    assert result.assignments.dtype.kind == "i"
    topic_ids = result.topic_ids
    assert len(topic_ids) >= 1
    # 키워드/라벨은 존재하는 토픽에 대해 채워져야 함
    for tid in topic_ids:
        assert tid in result.topic_keywords
        assert isinstance(result.topic_keywords[tid], list)
        assert tid in result.topic_labels
    assert result.runtime_sec >= 0
    assert result.config_hash == config.hash()
    assert result.seed == 42


def test_qualit_seed_only_changes_clustering(corpus, embeddings):
    """동일 설정·다른 시드에서 키프레이즈 추출(LLM)은 재호출되지만
    스텁이므로 결정적 — 구조 검증: 두 시드 모두 유효한 결과."""
    config = QualITConfig(n_topics=2, hallucination_filter="none", clustering="kmeans")
    llm = StubLLM()
    embedder = StubEmbedder(dim=32)
    r1 = QualITModel(config, llm=llm, embedder=embedder).fit(corpus, embeddings, seed=1)
    r2 = QualITModel(config, llm=llm, embedder=embedder).fit(corpus, embeddings, seed=2)
    assert len(r1.assignments) == len(r2.assignments) == len(corpus)


def test_presets_are_registered():
    for key, config in PRESETS.items():
        assert config.name in MODEL_REGISTRY, f"{key}의 모델 {config.name} 미등록"
        assert config.hash()  # 해시 가능


def test_config_hash_stability():
    a = QualITConfig()
    b = QualITConfig()
    c = QualITConfig(clustering="hdbscan")
    assert a.hash() == b.hash()
    assert a.hash() != c.hash()
