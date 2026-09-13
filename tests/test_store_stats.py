"""실험 저장소 라운드트립 + 통계 검정의 scipy 대조 검증."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from core.experiments import store
from core.experiments.stats import paired_tests, summarize
from core.types import RunRecord


def _make_record(experiment_id, model, display, seed, coherence, tmp_prefix=""):
    return RunRecord(
        run_id=f"{model}_{tmp_prefix}cfg_s{seed}",
        experiment_id=experiment_id,
        corpus_hash="testhash",
        corpus_name="test",
        regime="headlines",
        model_name=model,
        config={"name": model, "display_name": display},
        config_hash=f"{tmp_prefix}cfg",
        seed=seed,
        metrics={"coherence_emb": coherence, "npmi": coherence / 2},
        topics={"0": {"label": "t", "keywords": ["a"], "size": 3}},
        created_at="2026-07-09T00:00:00",
        status="done",
    )


def test_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("core.experiments.store.RUNS_DIR", tmp_path)
    rec = _make_record("exp1", "lda", "LDA", 42, 0.8)
    store.save_run(rec)
    assert store.run_exists("exp1", rec.run_id)
    loaded = store.load_run("exp1", rec.run_id)
    assert loaded is not None
    assert loaded.metrics["coherence_emb"] == 0.8
    assert loaded.config["display_name"] == "LDA"
    assert store.list_experiments()[0]["experiment_id"] == "exp1"


def test_summarize_mean_std():
    records = [
        _make_record("e", "lda", "LDA", s, v)
        for s, v in [(1, 0.7), (2, 0.8), (3, 0.9)]
    ]
    df = summarize(records)
    row = df[df["model"] == "LDA"].iloc[0]
    assert abs(row["coherence_emb_mean"] - 0.8) < 1e-9
    assert abs(row["coherence_emb_std"] - np.std([0.7, 0.8, 0.9], ddof=1)) < 1e-9


def test_paired_tests_matches_scipy():
    a_vals = [0.80, 0.85, 0.83, 0.88, 0.82]
    b_vals = [0.75, 0.78, 0.80, 0.79, 0.77]
    records = []
    for seed, (a, b) in enumerate(zip(a_vals, b_vals)):
        records.append(_make_record("e", "qualit", "QualIT", seed, a, "q"))
        records.append(_make_record("e", "bertopic", "BERTopic", seed, b, "b"))
    result = paired_tests(records, "QualIT", ["BERTopic"], "coherence_emb")
    row = result.iloc[0]
    t_ref, p_ref = sps.ttest_rel(a_vals, b_vals)
    assert row["n_pairs"] == 5
    assert row["t_stat"] == pytest.approx(float(t_ref))
    assert row["t_p"] == pytest.approx(float(p_ref))
    diff = np.array(a_vals) - np.array(b_vals)
    assert row["cohen_d"] == pytest.approx(float(diff.mean() / diff.std(ddof=1)))
    assert row["holm_p"] == pytest.approx(min(1.0, float(p_ref)))  # 비교 1건 → 보정 동일


def test_paired_tests_insufficient_pairs():
    records = [
        _make_record("e", "qualit", "QualIT", 1, 0.8, "q"),
        _make_record("e", "bertopic", "BERTopic", 2, 0.7, "b"),  # 시드 불일치
    ]
    result = paired_tests(records, "QualIT", ["BERTopic"], "coherence_emb")
    assert result.iloc[0]["n_pairs"] == 0
