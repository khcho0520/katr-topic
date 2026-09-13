"""탐색 세션(워크스페이스) 저장/복원.

페이지 2의 단일 실행 결과(설정 스냅샷 + 토픽 결과 + 지표)를 이름 붙여
runs/_workspaces/에 저장하고, 앱 재시작 후에도 목록에서 복원할 수 있게 한다.
(MCPanal의 workspace save/restore 패턴을 단순화해 이식)
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import RUNS_DIR
from core.types import MetricReport, TopicResult
from core.utils.jsonsafe import json_safe

WORKSPACES_DIR = RUNS_DIR / "_workspaces"

# 복원 불가능하거나 무거운 artifacts 항목은 저장에서 제외한다
_ARTIFACT_MAX_ITEMS = 20000


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w가-힣-]+", "_", name.strip()).strip("_")
    return slug or "session"


def save_workspace(
    name: str,
    corpus_hash: str,
    corpus_name: str,
    regime: str,
    config_dict: Dict[str, Any],
    seed: int,
    result: TopicResult,
    report: MetricReport,
) -> str:
    """세션 스냅샷 저장 → workspace_id 반환."""
    created_at = datetime.now()
    workspace_id = f"{_slugify(name)}_{created_at.strftime('%Y%m%d_%H%M%S')}"
    payload = {
        "workspace_id": workspace_id,
        "name": name.strip() or workspace_id,
        "created_at": created_at.isoformat(timespec="seconds"),
        "corpus_hash": corpus_hash,
        "corpus_name": corpus_name,
        "regime": regime,
        "config": json_safe(config_dict),
        "seed": int(seed),
        "result": {
            "model_name": result.model_name,
            "config_hash": result.config_hash,
            "seed": int(result.seed),
            "assignments": [int(a) for a in result.assignments],
            "topic_keywords": {str(k): list(v) for k, v in result.topic_keywords.items()},
            "topic_labels": {str(k): str(v) for k, v in result.topic_labels.items()},
            "runtime_sec": float(result.runtime_sec),
            "artifacts": json_safe(_trim_artifacts(result.artifacts)),
        },
        "metrics": json_safe(report.scalars),
        "per_topic": (
            report.per_topic.to_dict(orient="records")
            if isinstance(report.per_topic, pd.DataFrame)
            else []
        ),
    }
    WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)
    path = WORKSPACES_DIR / f"{workspace_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return workspace_id


def _trim_artifacts(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in (artifacts or {}).items():
        if isinstance(value, (list, tuple)) and len(value) > _ARTIFACT_MAX_ITEMS:
            continue
        out[key] = value
    return out


def list_workspaces() -> List[Dict[str, Any]]:
    """저장된 세션 목록 (최신순, 메타 정보만)."""
    out = []
    if not WORKSPACES_DIR.exists():
        return out
    for path in WORKSPACES_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out.append(
                {
                    "workspace_id": data["workspace_id"],
                    "name": data.get("name", path.stem),
                    "created_at": data.get("created_at", ""),
                    "corpus_hash": data.get("corpus_hash", ""),
                    "corpus_name": data.get("corpus_name", ""),
                    "regime": data.get("regime", ""),
                    "model": (data.get("config") or {}).get("display_name")
                    or (data.get("config") or {}).get("name", ""),
                    "seed": data.get("seed"),
                }
            )
        except Exception:
            continue
    return sorted(out, key=lambda w: w.get("created_at", ""), reverse=True)


def load_workspace(workspace_id: str) -> Optional[Dict[str, Any]]:
    """세션 복원: TopicResult/MetricReport 객체로 되돌려 반환."""
    path = WORKSPACES_DIR / f"{workspace_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    r = data["result"]
    result = TopicResult(
        model_name=r["model_name"],
        config_hash=r["config_hash"],
        seed=int(r["seed"]),
        assignments=np.array(r["assignments"], dtype=int),
        topic_keywords={int(k): list(v) for k, v in r["topic_keywords"].items()},
        topic_labels={int(k): str(v) for k, v in r["topic_labels"].items()},
        artifacts=r.get("artifacts") or {},
        runtime_sec=float(r.get("runtime_sec", 0.0)),
    )
    report = MetricReport(
        scalars={k: float(v) for k, v in (data.get("metrics") or {}).items()},
        per_topic=pd.DataFrame(data.get("per_topic") or []),
    )
    return {
        "workspace_id": data["workspace_id"],
        "name": data.get("name", workspace_id),
        "created_at": data.get("created_at", ""),
        "corpus_hash": data.get("corpus_hash", ""),
        "corpus_name": data.get("corpus_name", ""),
        "regime": data.get("regime", ""),
        "config": data.get("config") or {},
        "seed": data.get("seed"),
        "result": result,
        "report": report,
    }


def delete_workspace(workspace_id: str) -> bool:
    path = WORKSPACES_DIR / f"{workspace_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
