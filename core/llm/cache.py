"""sqlite 기반 범용 디스크 캐시. LLM 출력·임베딩을 저장해 반복실험 비용을 없앤다."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional


class DiskCache:
    """key(str) → JSON 값. 스레드 안전. 히트/미스 카운터로 캐시 검증 지원."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
            return json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, payload)
            )
            self._conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        with self._lock:
            for key in keys:
                row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
                if row is None:
                    self.misses += 1
                else:
                    self.hits += 1
                    out[key] = json.loads(row[0])
        return out

    def set_many(self, items: dict[str, Any]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                [(k, json.dumps(v, ensure_ascii=False)) for k, v in items.items()],
            )
            self._conn.commit()

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_caches: dict[str, DiskCache] = {}
_caches_lock = threading.Lock()


def get_cache(path: Path | str) -> DiskCache:
    key = str(path)
    with _caches_lock:
        if key not in _caches:
            _caches[key] = DiskCache(key)
        return _caches[key]
