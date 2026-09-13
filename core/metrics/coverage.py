"""커버리지·구조 지표: 할당률, 토픽 수, 토픽 크기 통계."""
from __future__ import annotations

from typing import Dict

import numpy as np


def compute_coverage_metrics(assignments: np.ndarray) -> Dict[str, float]:
    n = len(assignments)
    if n == 0:
        return {"coverage": 0.0, "n_topics_found": 0.0, "topic_size_std": 0.0}
    assigned = int(np.sum(assignments >= 0))
    topic_ids = sorted(t for t in set(int(a) for a in assignments) if t >= 0)
    sizes = np.array([int(np.sum(assignments == t)) for t in topic_ids]) if topic_ids else np.array([0])
    return {
        "coverage": assigned / n,
        "n_topics_found": float(len(topic_ids)),
        "topic_size_std": float(np.std(sizes)),
        "topic_size_min": float(sizes.min()),
        "topic_size_max": float(sizes.max()),
    }
