"""지표 sanity: 잘 분리된 클러스터 → 높은 점수, 뒤섞인 라벨 → 낮은 점수."""
from __future__ import annotations

import numpy as np

from core.metrics.coverage import compute_coverage_metrics
from core.metrics.embedding_metrics import (
    compute_embedding_metrics,
    topic_coherence,
    topic_separation,
)
from core.metrics.lexical_metrics import npmi_for_topic, topic_diversity
from tests.stubs import blob_embeddings


def test_embedding_metrics_separated_vs_shuffled():
    matrix, labels = blob_embeddings(20, dim=32, n_clusters=3)
    good = compute_embedding_metrics(matrix, labels)
    rng = np.random.default_rng(1)
    shuffled = rng.permutation(labels)
    bad = compute_embedding_metrics(matrix, shuffled)

    assert good["silhouette"] > 0.5
    assert good["silhouette"] > bad["silhouette"]
    assert good["coherence_emb"] > bad["coherence_emb"]
    assert good["separation"] > 0.1


def test_topic_coherence_bounds():
    tight = np.tile(np.array([1.0, 0.0, 0.0]), (5, 1)) + 0.01
    assert topic_coherence(tight) > 0.99
    assert topic_coherence(np.zeros((1, 3))) == 1.0


def test_topic_separation_orthogonal():
    centroids = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    assert abs(topic_separation(centroids) - 1.0) < 1e-9


def test_diversity():
    distinct = {0: ["가격", "부동산"], 1: ["교육", "학교"]}
    overlapping = {0: ["가격", "부동산"], 1: ["가격", "부동산"]}
    assert topic_diversity(distinct, top_k=2) == 1.0
    assert topic_diversity(overlapping, top_k=2) == 0.5


def test_npmi_cooccurring_pair_high():
    # 두 단어가 항상 같이 등장 → NPMI ≈ 1
    baseline = [{"인공", "지능"} for _ in range(10)] + [{"다른", "단어"} for _ in range(10)]
    high = npmi_for_topic(["인공", "지능"], baseline)
    assert high > 0.9
    # 독립적으로만 등장하는 쌍 → 쌍 df 부족으로 0
    baseline2 = [{"인공"} for _ in range(10)] + [{"지능"} for _ in range(10)]
    assert npmi_for_topic(["인공", "지능"], baseline2) == 0.0


def test_coverage_with_outliers():
    assignments = np.array([0, 0, 1, 1, -1, -1, -1, -1])
    m = compute_coverage_metrics(assignments)
    assert abs(m["coverage"] - 0.5) < 1e-9
    assert m["n_topics_found"] == 2
