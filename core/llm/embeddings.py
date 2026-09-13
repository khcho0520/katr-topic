"""bge-m3 임베딩 (Ollama /api/embed 배치) + sqlite 캐시.

모든 모델·지표가 이 임베더가 만든 동일한 행렬을 공유한다 (공정성 규칙).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

import numpy as np
import requests

from config import (
    DEFAULT_EMBED_MODEL,
    EMBED_BATCH_SIZE,
    EMBED_CACHE_PATH,
    EMBED_TIMEOUT,
    get_ollama_hosts,
)
from core.llm.cache import DiskCache, get_cache
from core.utils.hashing import text_hash
from core.utils.progress import ProgressCallback

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        hosts: Optional[List[str]] = None,
        cache: Optional[DiskCache] = None,
        batch_size: int = EMBED_BATCH_SIZE,
    ):
        self.model = model
        self.hosts = hosts or get_ollama_hosts()
        self.cache = cache if cache is not None else get_cache(EMBED_CACHE_PATH)
        self.batch_size = batch_size
        self.call_count = 0

    def _cache_key(self, text: str) -> str:
        return f"emb:{self.model}:{text_hash(text, 32)}"

    def _request_batch(self, texts: List[str]) -> List[List[float]]:
        """Ollama /api/embed 배치 호출. 지수 백오프 재시도, 호스트 폴백."""
        last_exc: Optional[Exception] = None
        for host in self.hosts:
            for attempt in range(3):
                try:
                    self.call_count += 1
                    r = requests.post(
                        f"{host}/api/embed",
                        json={"model": self.model, "input": texts},
                        timeout=EMBED_TIMEOUT,
                    )
                    r.raise_for_status()
                    embeddings = r.json().get("embeddings") or []
                    if len(embeddings) != len(texts):
                        raise RuntimeError(
                            f"embedding_count_mismatch:{len(embeddings)}!={len(texts)}"
                        )
                    return embeddings
                except Exception as e:
                    last_exc = e
                    logger.warning("임베딩 배치 실패 host=%s attempt=%d: %s", host, attempt + 1, e)
                    time.sleep(1.5 ** attempt)
        raise RuntimeError(f"모든 호스트에서 임베딩 실패: {last_exc}")

    def embed_texts(
        self,
        texts: List[str],
        progress: Optional[ProgressCallback] = None,
    ) -> np.ndarray:
        """텍스트 목록 → (n, dim) 임베딩 행렬. 캐시된 항목은 재계산하지 않는다."""
        n = len(texts)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        keys = [self._cache_key(t) for t in texts]
        cached = self.cache.get_many(keys)
        results: List[Optional[List[float]]] = [cached.get(k) for k in keys]

        # 캐시에 없는 것만 배치 임베딩 (동일 텍스트 중복은 1회만 호출)
        pending: dict[str, List[int]] = {}
        for i, vec in enumerate(results):
            if vec is None:
                pending.setdefault(texts[i], []).append(i)

        unique_texts = list(pending.keys())
        total_batches = (len(unique_texts) + self.batch_size - 1) // self.batch_size
        for bi in range(0, len(unique_texts), self.batch_size):
            batch = unique_texts[bi : bi + self.batch_size]
            # 빈 문자열은 API 오류를 피하기 위해 공백 1개로 대체
            embeddings = self._request_batch([t if t.strip() else " " for t in batch])
            to_cache = {}
            for t, vec in zip(batch, embeddings):
                to_cache[self._cache_key(t)] = vec
                for idx in pending[t]:
                    results[idx] = vec
            self.cache.set_many(to_cache)
            if progress:
                done = bi // self.batch_size + 1
                progress(done / max(1, total_batches), f"임베딩 {done}/{total_batches} 배치")

        matrix = np.array(results, dtype=np.float32)
        return matrix

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


_default_embedder: Optional[Embedder] = None
_default_lock = threading.Lock()


def get_default_embedder() -> Embedder:
    global _default_embedder
    with _default_lock:
        if _default_embedder is None:
            _default_embedder = Embedder()
        return _default_embedder
