"""탐색 세션(워크스페이스) 저장/복원 라운드트립 검증."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.experiments import workspace as ws
from core.types import MetricReport, TopicResult


def _sample_result() -> TopicResult:
    return TopicResult(
        model_name="QualIT (논문 재현)",
        config_hash="abc123def456",
        seed=42,
        assignments=np.array([0, 0, 1, 1, -1], dtype=int),
        topic_keywords={0: ["인공지능", "정책"], 1: ["부동산", "시장"]},
        topic_labels={0: "AI 정책", 1: "부동산 동향"},
        artifacts={"chosen_k": 2, "hallucination_removed_rate": np.float64(0.08)},
        runtime_sec=12.5,
    )


def _sample_report() -> MetricReport:
    return MetricReport(
        scalars={"coherence_emb": 0.71, "npmi": 0.42, "coverage": 0.8},
        per_topic=pd.DataFrame(
            [{"topic_id": 0, "label": "AI 정책", "size": 2, "coherence": 0.75, "keywords": "인공지능, 정책"}]
        ),
    )


def test_workspace_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACES_DIR", tmp_path)

    workspace_id = ws.save_workspace(
        name="테스트 세션 1차!",
        corpus_hash="corpushash123",
        corpus_name="sample",
        regime="headlines",
        config_dict={"name": "qualit", "display_name": "QualIT (논문 재현)", "clustering": "kmeans"},
        seed=42,
        result=_sample_result(),
        report=_sample_report(),
    )
    assert workspace_id.startswith("테스트_세션_1차")

    listing = ws.list_workspaces()
    assert len(listing) == 1
    assert listing[0]["model"] == "QualIT (논문 재현)"
    assert listing[0]["corpus_hash"] == "corpushash123"

    loaded = ws.load_workspace(workspace_id)
    assert loaded is not None
    result = loaded["result"]
    assert isinstance(result, TopicResult)
    assert list(result.assignments) == [0, 0, 1, 1, -1]
    assert result.topic_keywords[0] == ["인공지능", "정책"]
    assert result.topic_labels[1] == "부동산 동향"
    assert result.artifacts["hallucination_removed_rate"] == 0.08
    report = loaded["report"]
    assert report.scalars["coherence_emb"] == 0.71
    assert len(report.per_topic) == 1
    assert loaded["config"]["clustering"] == "kmeans"


def test_workspace_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACES_DIR", tmp_path)
    wid = ws.save_workspace(
        name="삭제할 세션", corpus_hash="h", corpus_name="c", regime="papers",
        config_dict={"name": "lda"}, seed=1,
        result=_sample_result(), report=_sample_report(),
    )
    assert ws.delete_workspace(wid) is True
    assert ws.load_workspace(wid) is None
    assert ws.list_workspaces() == []


def test_load_missing_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "WORKSPACES_DIR", tmp_path)
    assert ws.load_workspace("없는세션_20260101_000000") is None
