"""BERTopic 베이스라인: 공유 bge-m3 임베딩 + UMAP + HDBSCAN.

c-TF-IDF 키워드도 공유 Kiwi 토크나이저로 계산해 어휘 지표 비교를 공정하게 한다.
"""
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
class BERTopicConfig(ModelConfig):
    name: str = "bertopic"
    n_topics: Optional[int] = None       # None = HDBSCAN 자동 결정
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    min_cluster_size: int = 0            # 0 = max(3, n_docs // 20) 자동
    calculate_probabilities: bool = False
    # 전수 할당 변형: HDBSCAN 아웃라이어(-1)를 공유 임베딩 공간의 최근접 토픽
    # 센트로이드에 할당 (QualIT assign_all과 동일 규칙 → 커버리지 교란 제거)
    assign_outliers: bool = False


@register("bertopic", BERTopicConfig)
class BERTopicModel(TopicModel):
    requires_embeddings = True

    def fit(
        self,
        corpus: Corpus,
        embeddings: Optional[np.ndarray],
        seed: int,
        progress: Optional[ProgressCallback] = None,
    ) -> TopicResult:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP

        if embeddings is None:
            raise ValueError("BERTopic은 공유 임베딩 행렬이 필요합니다.")

        started = time.perf_counter()
        cfg: BERTopicConfig = self.config  # type: ignore[assignment]
        n_docs = len(corpus)
        texts = corpus.texts

        min_cluster_size = cfg.min_cluster_size or max(3, n_docs // 20)

        umap_model = UMAP(
            n_neighbors=min(cfg.umap_n_neighbors, max(2, n_docs - 1)),
            n_components=min(cfg.umap_n_components, max(2, n_docs - 2)),
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=False,
        )

        # c-TF-IDF 키워드를 공유 Kiwi 토큰으로 계산
        token_map = {}
        if corpus.tokenized is not None:
            token_map = {text: tokens for text, tokens in zip(texts, corpus.tokenized)}

        def _analyzer(doc: str):
            if doc in token_map:
                return token_map[doc]
            from core.data.tokenize_ko import tokenize

            return tokenize(doc)

        vectorizer_model = CountVectorizer(analyzer=_analyzer, lowercase=False)

        topic_model = BERTopic(
            language="multilingual",   # 기본 "english"는 전처리에서 한글을 제거함
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=cfg.n_topics,
            calculate_probabilities=cfg.calculate_probabilities,
            verbose=False,
        )
        if progress:
            progress(0.1, "BERTopic 학습 (UMAP+HDBSCAN)")
        topics, _probs = topic_model.fit_transform(texts, embeddings=np.asarray(embeddings, dtype=np.float64))
        assignments = np.array(topics, dtype=int)
        n_outliers_before = int(np.sum(assignments == -1))

        if cfg.assign_outliers and n_outliers_before:
            emb = np.asarray(embeddings, dtype=np.float32)
            topic_ids_found = sorted(t for t in set(int(t) for t in assignments) if t >= 0)
            if topic_ids_found:
                centroids = np.stack([emb[assignments == t].mean(axis=0) for t in topic_ids_found])
                c_norm = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
                outlier_idx = np.where(assignments == -1)[0]
                d = emb[outlier_idx]
                d_norm = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
                nearest = np.argmax(d_norm @ c_norm.T, axis=1)
                for pos, doc_idx in enumerate(outlier_idx):
                    assignments[doc_idx] = topic_ids_found[int(nearest[pos])]

        keywords = {}
        labels = {}
        for tid in sorted(set(int(t) for t in assignments)):
            if tid == -1:
                continue
            topic_terms = topic_model.get_topic(tid) or []
            kws = [str(term) for term, _w in topic_terms[:TOP_KEYWORDS] if str(term).strip()]
            keywords[tid] = kws
            labels[tid] = ", ".join(kws[:4])

        if progress:
            progress(1.0, "BERTopic 완료")
        return TopicResult(
            model_name=cfg.label,
            config_hash=cfg.hash(),
            seed=seed,
            assignments=assignments,
            topic_keywords=keywords,
            topic_labels=labels,
            artifacts={
                "min_cluster_size": min_cluster_size,
                "n_outliers_before_assign": n_outliers_before,
            },
            runtime_sec=time.perf_counter() - started,
        )
