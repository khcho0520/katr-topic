"""준비된 Corpus의 디스크 영속화: cache/corpora/{hash}/ (parquet + meta JSON).

페이지 간 재파싱 없이 로드하고, 실험 기록의 corpus_hash 재현성을 보장한다.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from config import CORPORA_DIR
from core.types import Corpus, Document


def save_corpus(corpus: Corpus) -> Path:
    target = CORPORA_DIR / corpus.content_hash
    target.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "doc_id": [d.doc_id for d in corpus.docs],
            "text": [d.text for d in corpus.docs],
            "source_id": [d.source_id for d in corpus.docs],
            "meta": [json.dumps(d.meta, ensure_ascii=False) for d in corpus.docs],
        }
    )
    df.to_parquet(target / "docs.parquet", index=False)

    if corpus.tokenized is not None:
        pd.DataFrame({"tokens": [json.dumps(t, ensure_ascii=False) for t in corpus.tokenized]}).to_parquet(
            target / "tokens.parquet", index=False
        )

    meta = {
        "regime": corpus.regime,
        "name": corpus.name,
        "content_hash": corpus.content_hash,
        "n_docs": len(corpus.docs),
        "meta": corpus.meta,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_corpus(content_hash: str) -> Corpus:
    target = CORPORA_DIR / content_hash
    meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(target / "docs.parquet")
    docs = [
        Document(
            doc_id=row.doc_id,
            text=row.text,
            source_id=row.source_id,
            meta=json.loads(row.meta) if row.meta else {},
        )
        for row in df.itertuples(index=False)
    ]
    tokenized = None
    tokens_path = target / "tokens.parquet"
    if tokens_path.exists():
        tdf = pd.read_parquet(tokens_path)
        tokenized = [json.loads(t) for t in tdf["tokens"]]
    return Corpus(
        regime=meta["regime"],
        docs=docs,
        content_hash=meta["content_hash"],
        name=meta.get("name", ""),
        tokenized=tokenized,
        meta=meta.get("meta", {}),
    )


def save_embeddings(content_hash: str, matrix: np.ndarray) -> Path:
    """코퍼스에 대응하는 임베딩 행렬을 npy로 저장 (외부에서 가져온 임베딩 재사용용)."""
    target = CORPORA_DIR / content_hash
    target.mkdir(parents=True, exist_ok=True)
    path = target / "embeddings.npy"
    np.save(path, np.asarray(matrix, dtype=np.float32))
    return path


def load_embeddings(content_hash: str) -> Optional[np.ndarray]:
    path = CORPORA_DIR / content_hash / "embeddings.npy"
    if not path.exists():
        return None
    return np.load(path)


def list_corpora() -> List[dict]:
    out = []
    if not CORPORA_DIR.exists():
        return out
    for entry in sorted(CORPORA_DIR.iterdir()):
        meta_path = entry / "meta.json"
        if meta_path.exists():
            try:
                out.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:
                continue
    return sorted(out, key=lambda m: m.get("saved_at", ""), reverse=True)


def delete_corpus(content_hash: str) -> bool:
    target = CORPORA_DIR / content_hash
    if target.exists():
        shutil.rmtree(target)
        return True
    return False


def corpus_exists(content_hash: str) -> bool:
    return (CORPORA_DIR / content_hash / "meta.json").exists()
