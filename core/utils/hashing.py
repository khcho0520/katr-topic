"""안정적 해시 유틸: 설정/코퍼스를 캐시 키와 run ID로 변환한다."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any


def _canonical(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"__class__": type(obj).__name__, **{k: _canonical(v) for k, v in dataclasses.asdict(obj).items()}}
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def stable_hash(obj: Any, length: int = 12) -> str:
    """dataclass/dict/list를 정규화된 JSON으로 직렬화해 sha256 접두어를 반환."""
    payload = json.dumps(_canonical(obj), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def text_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def texts_hash(texts: list[str], length: int = 12) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:length]
