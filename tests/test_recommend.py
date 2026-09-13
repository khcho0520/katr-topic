"""공통 조건 자동 제안 알고리즘 검증."""
from __future__ import annotations

import numpy as np

from core.experiments.recommend import recommend_common_conditions
from core.types import Corpus, Document
from tests.stubs import blob_embeddings


def _corpus_of(n: int, regime: str = "headlines", n_sources: int = 1) -> Corpus:
    docs = [
        Document(doc_id=f"d{i}", text=f"문서 {i}", source_id=f"s{i % n_sources}")
        for i in range(n)
    ]
    return Corpus(regime=regime, docs=docs, content_hash="testhash", name="t")


def test_recovers_true_cluster_count_on_blobs():
    matrix, _ = blob_embeddings(80, dim=32, n_clusters=5, spread=0.03)
    corpus = _corpus_of(len(matrix))
    reco = recommend_common_conditions(corpus, matrix)
    # 뚜렷한 5블롭 → 관용 규칙으로도 5 부근을 잡아야 함
    assert 4 <= reco.n_topics <= 6, f"K={reco.n_topics}, scores={reco.k_scores}"
    assert reco.rationale


def test_deterministic():
    matrix, _ = blob_embeddings(60, dim=16, n_clusters=3)
    corpus = _corpus_of(len(matrix))
    r1 = recommend_common_conditions(corpus, matrix)
    r2 = recommend_common_conditions(corpus, matrix)
    assert r1.n_topics == r2.n_topics
    assert r1.bertopic_min_cluster_size == r2.bertopic_min_cluster_size
    assert r1.k_scores == r2.k_scores


def test_mcs_and_llm_docs_scale_with_corpus():
    matrix, _ = blob_embeddings(100, dim=16, n_clusters=4)
    corpus = _corpus_of(len(matrix))
    reco = recommend_common_conditions(corpus, matrix)
    n, k = len(corpus), reco.n_topics
    assert reco.bertopic_min_cluster_size == max(3, min(500, n // (k * 20)))
    assert reco.qualit_max_llm_docs <= max(200, n)
    assert reco.qualit_max_llm_docs >= min(n, 200)


def test_papers_regime_caps_k_by_source_count():
    # 원문 2편짜리 논문 코퍼스 → K 상한 = max(4, 2×2) = 4
    matrix, _ = blob_embeddings(60, dim=16, n_clusters=6, spread=0.03)
    corpus = _corpus_of(len(matrix), regime="papers", n_sources=2)
    reco = recommend_common_conditions(corpus, matrix)
    assert reco.n_topics <= 4
    assert any("논문 체제" in line for line in reco.rationale)


def test_small_corpus_llm_docs_covers_all():
    matrix, _ = blob_embeddings(30, dim=16, n_clusters=2)
    corpus = _corpus_of(len(matrix))
    reco = recommend_common_conditions(corpus, matrix)
    assert reco.qualit_max_llm_docs == len(corpus)
