"""외적 타당도(NMI/ARI/Purity)와 시드 간 안정성(ARI) 지표 검증."""
from __future__ import annotations

import numpy as np
import pytest

from core.experiments.stats import stability_table
from core.metrics.external import (
    compute_external_metrics,
    extract_labels,
    extract_major_category,
)
from core.types import Corpus, Document, RunRecord


def _labeled_corpus(categories):
    docs = [
        Document(doc_id=f"d{i}", text=f"문서 {i}", source_id="s",
                 meta={"category": c} if c else {})
        for i, c in enumerate(categories)
    ]
    return Corpus(regime="headlines", docs=docs, content_hash="h", name="t")


def test_extract_major_category():
    assert extract_major_category("사회>사건_사고 | 사회>여성") == "사회"
    assert extract_major_category("경제>부동산") == "경제"
    assert extract_major_category("정치") == "정치"
    assert extract_major_category("") == ""


def test_extract_labels_threshold():
    # 레이블 비율 50% 미만 → None
    corpus = _labeled_corpus(["사회"] * 5 + [None] * 15)
    assert extract_labels(corpus) is None
    corpus2 = _labeled_corpus(["사회>사건"] * 12 + ["경제>금융"] * 8)
    labels = extract_labels(corpus2)
    assert labels == ["사회"] * 12 + ["경제"] * 8


def test_external_metrics_perfect_alignment():
    labels = ["사회"] * 20 + ["경제"] * 20
    assignments = np.array([0] * 20 + [1] * 20)
    m = compute_external_metrics(assignments, labels)
    assert m["nmi"] == pytest.approx(1.0)
    assert m["ari"] == pytest.approx(1.0)
    assert m["purity"] == pytest.approx(1.0)
    assert m["label_coverage"] == pytest.approx(1.0)
    assert m["n_label_classes"] == 2


def test_external_metrics_random_low():
    rng = np.random.default_rng(0)
    labels = (["사회"] * 50 + ["경제"] * 50)
    assignments = rng.integers(0, 2, size=100)
    m = compute_external_metrics(assignments, labels)
    assert m["nmi"] < 0.2
    assert abs(m["ari"]) < 0.2


def test_external_metrics_excludes_outliers():
    labels = ["사회"] * 10 + ["경제"] * 10
    assignments = np.array([0] * 10 + [-1] * 10)   # 경제 문서 전부 미할당
    m = compute_external_metrics(assignments, labels)
    # 남은 유효 표본의 레이블/토픽이 각 1종 → 계산 불가로 빈 dict
    assert m == {}


def _record(model, seed, assignments, config_hash="cfg"):
    return RunRecord(
        run_id=f"{model}_{config_hash}_s{seed}", experiment_id="e",
        corpus_hash="h", corpus_name="t", regime="headlines",
        model_name=model, config={"name": model, "display_name": model.upper()},
        config_hash=config_hash, seed=seed, metrics={}, topics={},
        created_at="2026-07-09T00:00:00", status="done",
        assignments=list(assignments),
    )


def test_stability_identical_and_permuted():
    base = [0] * 20 + [1] * 20
    permuted = [7] * 20 + [3] * 20        # 같은 분할, 다른 토픽 ID → ARI 1
    records = [
        _record("lda", 1, base),
        _record("lda", 2, base),
        _record("lda", 3, permuted),
    ]
    df = stability_table(records)
    row = df.iloc[0]
    assert row["n_pairs"] == 3
    assert row["stability_ari_mean"] == pytest.approx(1.0)


def test_stability_unstable_low():
    rng = np.random.default_rng(0)
    records = [
        _record("qualit", s, rng.integers(0, 5, size=200)) for s in range(3)
    ]
    df = stability_table(records)
    assert df.iloc[0]["stability_ari_mean"] < 0.2


def test_stability_skips_records_without_assignments():
    records = [
        _record("lda", 1, []),
        _record("lda", 2, []),
    ]
    assert stability_table(records).empty


def test_runrecord_backward_compat():
    """assignments 없는 과거 JSON도 로드 가능해야 함."""
    data = {
        "run_id": "r", "experiment_id": "e", "corpus_hash": "h",
        "corpus_name": "t", "regime": "headlines", "model_name": "lda",
        "config": {}, "config_hash": "c", "seed": 1, "metrics": {},
        "topics": {}, "created_at": "2026-01-01T00:00:00", "status": "done",
        "error": "",
    }
    record = RunRecord(**data)
    assert record.assignments == []
