"""단일 공정 평가 진입점: 모든 모델의 TopicResult를 동일 기준으로 평가한다."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.metrics.coverage import compute_coverage_metrics
from core.metrics.embedding_metrics import compute_embedding_metrics, per_topic_coherence
from core.metrics.external import compute_external_metrics, extract_labels
from core.metrics.lexical_metrics import compute_lexical_metrics
from core.types import Corpus, MetricReport, TopicResult


def evaluate(
    result: TopicResult,
    corpus: Corpus,
    embeddings: np.ndarray,
    judge: bool = False,
    judge_model: Optional[str] = None,
) -> MetricReport:
    scalars: dict[str, float] = {}
    scalars.update(compute_embedding_metrics(embeddings, result.assignments))
    if corpus.tokenized is not None:
        scalars.update(compute_lexical_metrics(result.topic_keywords, corpus.tokenized))
    scalars.update(compute_coverage_metrics(result.assignments))
    scalars["runtime_sec"] = float(result.runtime_sec)

    # 외적 타당도: 카테고리 레이블이 충분한 코퍼스(BigKinds 등)에서만 자동 수행
    labels = extract_labels(corpus)
    if labels is not None:
        scalars.update(compute_external_metrics(result.assignments, labels))

    if judge:
        from config import DEFAULT_JUDGE_MODEL
        from core.metrics.llm_judge import judge_topics

        scalars.update(judge_topics(result, corpus, model=judge_model or DEFAULT_JUDGE_MODEL))

    coherences = per_topic_coherence(embeddings, result.assignments)
    sizes = result.topic_sizes()
    per_topic = pd.DataFrame(
        [
            {
                "topic_id": tid,
                "label": result.topic_labels.get(tid, ""),
                "size": sizes.get(tid, 0),
                "coherence": coherences.get(tid, float("nan")),
                "keywords": ", ".join((result.topic_keywords.get(tid) or [])[:10]),
            }
            for tid in result.topic_ids
        ]
    )
    return MetricReport(scalars=scalars, per_topic=per_topic)
