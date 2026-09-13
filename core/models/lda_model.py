"""LDA 베이스라인 (Kiwi 토큰 기반 CountVectorizer + sklearn LDA)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.models.base import ModelConfig, TopicModel, register
from core.types import Corpus, TopicResult
from core.utils.progress import ProgressCallback

TOP_KEYWORDS = 10


@dataclass(frozen=True)
class LDAConfig(ModelConfig):
    name: str = "lda"
    n_topics: Optional[int] = 10
    max_features: int = 2000
    max_iter: int = 20


def _fit_matrix_model(corpus: Corpus, vectorizer, decomposer):
    """빈 토큰 문서를 제외하고 학습, 원본 인덱스 매핑을 함께 반환."""
    if corpus.tokenized is None:
        raise ValueError("corpus.tokenized가 없습니다. 코퍼스 준비 단계에서 토큰화를 먼저 수행하세요.")
    valid_indices = [i for i, toks in enumerate(corpus.tokenized) if toks]
    token_docs = [corpus.tokenized[i] for i in valid_indices]
    if len(token_docs) < 2:
        raise ValueError("토큰화된 문서가 2건 미만입니다.")
    matrix = vectorizer.fit_transform(token_docs)
    doc_topic = decomposer.fit_transform(matrix)
    return doc_topic, valid_indices, vectorizer, decomposer


def _extract_component_keywords(decomposer, vectorizer, top_k: int = TOP_KEYWORDS):
    terms = np.array(vectorizer.get_feature_names_out())
    keywords = {}
    for tid, weights in enumerate(decomposer.components_):
        top = np.argsort(weights)[::-1][:top_k]
        keywords[tid] = [str(terms[i]) for i in top if weights[i] > 0]
    return keywords


def _build_result(config, corpus, doc_topic, valid_indices, keywords, seed, started) -> TopicResult:
    n_docs = len(corpus)
    assignments = np.full(n_docs, -1, dtype=int)
    dist = np.zeros((n_docs, doc_topic.shape[1]), dtype=np.float64)
    argmax = np.argmax(doc_topic, axis=1)
    for pos, original in enumerate(valid_indices):
        assignments[original] = int(argmax[pos])
        dist[original] = doc_topic[pos]
    labels = {tid: ", ".join(kws[:4]) for tid, kws in keywords.items()}
    return TopicResult(
        model_name=config.label,
        config_hash=config.hash(),
        seed=seed,
        assignments=assignments,
        topic_keywords=keywords,
        topic_labels=labels,
        doc_topic_dist=dist,
        runtime_sec=time.perf_counter() - started,
    )


@register("lda", LDAConfig)
class LDAModel(TopicModel):
    requires_embeddings = False

    def fit(
        self,
        corpus: Corpus,
        embeddings: Optional[np.ndarray],
        seed: int,
        progress: Optional[ProgressCallback] = None,
    ) -> TopicResult:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer

        started = time.perf_counter()
        cfg: LDAConfig = self.config  # type: ignore[assignment]
        n_topics = cfg.n_topics or 10

        vectorizer = CountVectorizer(
            max_features=cfg.max_features,
            min_df=1,
            analyzer=lambda tokens: tokens,   # 이미 Kiwi 토큰화된 입력
            lowercase=False,
        )
        # 어휘 크기보다 토픽 수가 많으면 축소
        if corpus.tokenized is None:
            raise ValueError("corpus.tokenized가 없습니다.")
        vocab_estimate = len({t for toks in corpus.tokenized for t in toks})
        n_topics = max(2, min(n_topics, vocab_estimate - 1)) if vocab_estimate > 2 else 2

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=seed,
            max_iter=cfg.max_iter,
            learning_method="online",
            learning_offset=50.0,
        )
        if progress:
            progress(0.1, f"LDA 학습 (K={n_topics})")
        doc_topic, valid_indices, vectorizer, lda = _fit_matrix_model(corpus, vectorizer, lda)
        keywords = _extract_component_keywords(lda, vectorizer)
        if progress:
            progress(1.0, "LDA 완료")
        return _build_result(cfg, corpus, doc_topic, valid_indices, keywords, seed, started)
