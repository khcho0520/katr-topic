"""실험 기록 영속화: runs/{experiment_id}/{run_id}.json + 통합 인덱스.

run_id가 이미 존재하면 러너가 스킵(재개)한다 — 크래시/중단 후 이어서 실행 가능.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import RUNS_DIR
from core.types import RunRecord


def _run_path(experiment_id: str, run_id: str) -> Path:
    return RUNS_DIR / experiment_id / f"{run_id}.json"


def save_run(record: RunRecord) -> Path:
    path = _run_path(record.experiment_id, record.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dataclasses.asdict(record), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def run_exists(experiment_id: str, run_id: str) -> bool:
    return _run_path(experiment_id, run_id).exists()


def load_run(experiment_id: str, run_id: str) -> Optional[RunRecord]:
    path = _run_path(experiment_id, run_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunRecord(**data)


def load_experiment_runs(experiment_id: str) -> List[RunRecord]:
    exp_dir = RUNS_DIR / experiment_id
    if not exp_dir.exists():
        return []
    records = []
    for path in sorted(exp_dir.glob("*.json")):
        try:
            records.append(RunRecord(**json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return records


def list_experiments() -> List[dict]:
    """실험 목록 요약 (results 페이지 브라우저용)."""
    out = []
    if not RUNS_DIR.exists():
        return out
    for exp_dir in sorted(RUNS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        runs = list(exp_dir.glob("*.json"))
        if not runs:
            continue
        try:
            sample = json.loads(runs[0].read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append(
            {
                "experiment_id": exp_dir.name,
                "n_runs": len(runs),
                "corpus_name": sample.get("corpus_name", ""),
                "regime": sample.get("regime", ""),
                "created_at": sample.get("created_at", ""),
            }
        )
    return sorted(out, key=lambda e: e.get("created_at", ""), reverse=True)


def runs_to_dataframe(records: List[RunRecord]) -> pd.DataFrame:
    """RunRecord → 지표를 열로 펼친 long-wide DataFrame."""
    rows = []
    for r in records:
        row = {
            "run_id": r.run_id,
            "experiment_id": r.experiment_id,
            "model": r.config.get("display_name") or r.model_name,
            "model_key": r.model_name,
            "config_hash": r.config_hash,
            "seed": r.seed,
            "corpus_name": r.corpus_name,
            "regime": r.regime,
            "status": r.status,
        }
        row.update(r.metrics)
        rows.append(row)
    return pd.DataFrame(rows)
