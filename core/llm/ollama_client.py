"""Ollama 멀티호스트 폴백 클라이언트 (MCPanal utils/ollama_client.py의 슬림 이식).

- OLLAMA_HOSTS 순서대로 시도 (Tailscale 원격 우선 → 로컬 폴백)
- 빈 응답 재시도, 대형 모델 타임아웃 연장
- Streamlit 의존성 없음. LLM 캐시와 호출 카운터 내장.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from config import (
    HEAVY_MODEL_MARKERS,
    LLM_CACHE_PATH,
    LLM_TIMEOUT_HEAVY,
    LLM_TIMEOUT_LIGHT,
    PROMPT_VERSION,
    get_ollama_hosts,
)
from core.llm.cache import DiskCache, get_cache
from core.utils.hashing import stable_hash

logger = logging.getLogger(__name__)


def _chat_timeout(model_name: str) -> int:
    model = (model_name or "").lower()
    if any(marker in model for marker in HEAVY_MODEL_MARKERS):
        return LLM_TIMEOUT_HEAVY
    return LLM_TIMEOUT_LIGHT


class OllamaClient:
    """chat_text 하나로 통일된 호출 계약. 캐시 우선, 실패 시 호스트 폴백."""

    def __init__(self, hosts: Optional[List[str]] = None, cache: Optional[DiskCache] = None):
        self.hosts = hosts or get_ollama_hosts()
        self.cache = cache if cache is not None else get_cache(LLM_CACHE_PATH)
        self.call_count = 0            # 실제 HTTP 호출 수 (캐시 히트 제외)
        self._lock = threading.Lock()

    # ---- 상태 확인 ----
    def ping(self) -> Optional[str]:
        """응답하는 첫 호스트 URL을 반환, 전부 실패하면 None."""
        for host in self.hosts:
            try:
                r = requests.get(f"{host}/api/tags", timeout=5)
                if r.status_code == 200:
                    return host
            except Exception:
                continue
        return None

    def list_models(self) -> List[str]:
        for host in self.hosts:
            try:
                r = requests.get(f"{host}/api/tags", timeout=5)
                r.raise_for_status()
                models = [m.get("name", "") for m in r.json().get("models", [])]
                return sorted(m for m in models if m)
            except Exception:
                continue
        return []

    # ---- 채팅 ----
    def chat_text(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        num_ctx: int = 16384,
        use_cache: bool = True,
        cache_tag: str = "",
    ) -> str:
        """첫 성공 응답의 content 텍스트를 반환. use_cache=True면 동일 요청은 캐시 재사용."""
        cache_key = ""
        if use_cache:
            cache_key = "chat:" + stable_hash(
                {
                    "v": PROMPT_VERSION,
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tag": cache_tag,
                },
                length=32,
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return str(cached)

        text = self._chat_uncached(model, messages, temperature, max_tokens, num_ctx)
        if use_cache and text:
            self.cache.set(cache_key, text)
        return text

    def _chat_uncached(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        num_ctx: int,
    ) -> str:
        timeout = _chat_timeout(model)
        last_exc: Optional[Exception] = None
        for host in self.hosts:
            is_remote = "localhost" not in host and "127.0.0.1" not in host
            attempts = 1 if is_remote else 2
            for attempt in range(attempts):
                try:
                    with self._lock:
                        self.call_count += 1
                    payload: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "think": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            "num_ctx": num_ctx,
                        },
                    }
                    started = time.monotonic()
                    r = requests.post(f"{host}/api/chat", json=payload, timeout=(10, timeout))
                    r.raise_for_status()
                    content = str(((r.json().get("message") or {}).get("content")) or "").strip()
                    if not content:
                        raise RuntimeError("Ollama 응답 content가 비어 있습니다.")
                    logger.debug(
                        "Ollama chat 성공 host=%s model=%s len=%d elapsed=%.1fs",
                        host, model, len(content), time.monotonic() - started,
                    )
                    return content
                except Exception as e:
                    logger.warning("Ollama chat 실패 host=%s model=%s: %s", host, model, e)
                    last_exc = e
                    if "비어 있습니다" in str(e) and attempt + 1 < attempts:
                        time.sleep(0.5)
                        continue
                    break
        raise RuntimeError(f"모든 Ollama 호스트 실패: {last_exc}")


_default_client: Optional[OllamaClient] = None
_default_lock = threading.Lock()


def get_default_client() -> OllamaClient:
    global _default_client
    with _default_lock:
        if _default_client is None:
            _default_client = OllamaClient()
        return _default_client
