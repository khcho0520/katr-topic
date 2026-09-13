"""MCPanal 경기도 언론분석의 월별 '대한민국 전체' 임베딩 세션 가져오기.

- 레지스트리: {MCPANAL_DIR}/data/bigkinds_workspaces.json
- 실데이터: ChromaDB(CHROMA_PATH)의 session_{id} 컬렉션 — 헤드라인 1건=문서 1건,
  bge-m3 1024차원 임베딩이 이미 저장돼 있어 **재임베딩 없이** 그대로 재사용한다.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.types import Corpus, Document
from core.utils.hashing import texts_hash

logger = logging.getLogger(__name__)

MCPANAL_DIR = Path(os.environ.get("TOPICNEW_MCPANAL_DIR", "/Volumes/project/MCPanal"))
CHROMA_PATH = os.environ.get("TOPICNEW_CHROMA_PATH", "/Volumes/project/ChromaDB")
REGISTRY_PATH = MCPANAL_DIR / "data" / "bigkinds_workspaces.json"


def registry_available() -> bool:
    return REGISTRY_PATH.exists()


def list_monthly_sessions() -> List[Dict[str, Any]]:
    """임베딩 완료된 월별 '대한민국 전체' 세션 목록 (기간순, 중복 세션 제거)."""
    if not registry_available():
        return []
    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("BigKinds workspace registry 로드 실패: %s", exc)
        return []

    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for workspace_id, workspace in (registry or {}).items():
        dataset_archive = workspace.get("dataset_archive") if isinstance(workspace, dict) else None
        if not isinstance(dataset_archive, dict):
            continue
        for dataset_key, entry in dataset_archive.items():
            if not isinstance(entry, dict):
                continue
            original = entry.get("original_dataset") or {}
            embedded = entry.get("embedded_dataset") or {}
            if original.get("region_name") != "대한민국 전체" or not embedded.get("embedded"):
                continue
            session_id = str(embedded.get("session_id") or "").strip()
            start_date = str(original.get("start_date") or "").strip()
            end_date = str(original.get("end_date") or "").strip()
            if not (session_id and start_date and end_date):
                continue
            source = {
                "session_id": session_id,
                "period_label": f"{start_date}~{end_date}",
                "start_date": start_date,
                "end_date": end_date,
                "document_count": int(embedded.get("headline_count") or original.get("headline_count") or 0),
                "workspace_id": str(workspace_id),
                "collection_name": (embedded.get("embed_result") or {}).get("collection_name")
                or f"session_{session_id}",
            }
            key = (start_date, end_date, session_id)
            current = by_key.get(key)
            # 누적 아카이브 workspace 항목 우선 (MCPanal과 동일 규칙)
            if current is None or source["workspace_id"] == "gyeonggi_news_timeseries_archive":
                by_key[key] = source
    return sorted(by_key.values(), key=lambda s: s["start_date"])


def _fetch_collection(collection_name: str) -> Tuple[List[str], List[dict], np.ndarray]:
    """Chroma 컬렉션에서 (헤드라인, 메타, 임베딩) 전체를 배치로 읽는다."""
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(collection_name)
    total = collection.count()
    texts: List[str] = []
    metas: List[dict] = []
    embeddings: List[np.ndarray] = []
    batch = 2000
    for offset in range(0, total, batch):
        result = collection.get(
            limit=batch, offset=offset, include=["documents", "metadatas", "embeddings"]
        )
        texts.extend(result["documents"])
        metas.extend(result["metadatas"])
        embeddings.append(np.asarray(result["embeddings"], dtype=np.float32))
    matrix = np.vstack(embeddings) if embeddings else np.zeros((0, 0), dtype=np.float32)
    return texts, metas, matrix


def load_monthly_corpus(
    sessions: List[Dict[str, Any]],
    name: str = "",
    dedup: bool = True,
    min_length: int = 5,
    progress=None,
) -> Tuple[Corpus, np.ndarray]:
    """선택한 월별 세션(들)을 하나의 헤드라인 Corpus로 병합.

    반환된 임베딩은 Chroma에 저장된 bge-m3 벡터 그대로 — Ollama 호출 없음.
    """
    if not sessions:
        raise ValueError("가져올 세션을 선택하세요.")

    all_texts: List[str] = []
    all_metas: List[dict] = []
    all_vectors: List[np.ndarray] = []
    for i, source in enumerate(sessions):
        if progress:
            progress(i / len(sessions), f"Chroma 로드: {source['period_label']}")
        texts, metas, matrix = _fetch_collection(source["collection_name"])
        if len(texts) != len(matrix):
            raise RuntimeError(
                f"{source['collection_name']}: 문서/임베딩 수 불일치 {len(texts)}/{len(matrix)}"
            )
        all_texts.extend(texts)
        all_metas.extend(metas)
        all_vectors.append(matrix)

    matrix = np.vstack(all_vectors)
    docs: List[Document] = []
    keep_vectors: List[np.ndarray] = []
    seen: set[str] = set()
    for idx, (text, meta) in enumerate(zip(all_texts, all_metas)):
        headline = (text or "").strip()
        if len(headline) < min_length:
            continue
        if dedup:
            if headline in seen:
                continue
            seen.add(headline)
        docs.append(
            Document(
                doc_id=f"h{len(docs)}",
                text=headline,
                source_id=str((meta or {}).get("session_id") or "mcpanal"),
                meta={
                    "date": str((meta or {}).get("date") or "")[:10],
                    "press": str((meta or {}).get("media") or ""),
                    "category": str((meta or {}).get("category") or ""),
                },
            )
        )
        keep_vectors.append(matrix[idx])

    if not docs:
        raise ValueError("유효한 헤드라인이 없습니다.")

    periods = [s["period_label"] for s in sessions]
    corpus = Corpus(
        regime="headlines",
        docs=docs,
        content_hash=texts_hash([d.text for d in docs]),
        name=name or f"bigkinds_{periods[0][:7]}" + (f"_{len(periods)}개월" if len(periods) > 1 else ""),
        meta={
            "source": "mcpanal_bigkinds_monthly",
            "periods": periods,
            "session_ids": [s["session_id"] for s in sessions],
            "n_docs": len(docs),
            "dedup": dedup,
            "min_length": min_length,
        },
    )
    if progress:
        progress(1.0, f"병합 완료: {len(docs)}건")
    return corpus, np.vstack(keep_vectors)
