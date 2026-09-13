"""NMF 베이스라인 (Kiwi 토큰 기반 TF-IDF + sklearn NMF)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.models.base import ModelConfig, TopicModel, register
from core.models.lda_model import _build_result, _extract_component_keywords, _fit_matrix_model
from core.types import Corpus, TopicResult
from core.utils.progress import ProgressCallback


@dataclass(frozen=True)
class NMFConfig(ModelConfig):
    name: str = "nmf"
    n_topics: Optional[int] = 10
    max_features: int = 2000
    max_iter: int = 300
    # 과한 정규화는 소규모 코퍼스에서 성분을 전부 0으로 만들어 토픽이 붕괴함
    alpha: float = 0.0
    l1_ratio: float = 0.5


@register("nmf", NMFConfig)
class NMFModel(TopicModel):
    requires_embeddings = False

    def fit(
        self,
        corpus: Corpus,
        embeddings: Optional[np.ndarray],
        seed: int,
        progress: Optional[ProgressCallback] = None,
    ) -> TopicResult:
        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfVectorizer

        started = time.perf_counter()
        cfg: NMFConfig = self.config  # type: ignore[assignment]
        n_topics = cfg.n_topics or 10

        if corpus.tokenized is None:
            raise ValueError("corpus.tokenized가 없습니다.")
        vocab_estimate = len({t for toks in corpus.tokenized for t in toks})
        n_topics = max(2, min(n_topics, vocab_estimate - 1)) if vocab_estimate > 2 else 2

        vectorizer = TfidfVectorizer(
            max_features=cfg.max_features,
            min_df=1,
            analyzer=lambda tokens: tokens,
            lowercase=False,
        )
        nmf = NMF(
            n_components=n_topics,
            random_state=seed,
            max_iter=cfg.max_iter,
            init="nndsvda",
            alpha_W=cfg.alpha,
            alpha_H=cfg.alpha,
            l1_ratio=cfg.l1_ratio,
        )
        if progress:
            progress(0.1, f"NMF 학습 (K={n_topics})")
        doc_topic, valid_indices, vectorizer, nmf = _fit_matrix_model(corpus, vectorizer, nmf)
        keywords = _extract_component_keywords(nmf, vectorizer)
        if progress:
            progress(1.0, "NMF 완료")
        return _build_result(cfg, corpus, doc_topic, valid_indices, keywords, seed, started)
