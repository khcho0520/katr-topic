"""1단계: LLM 키프레이즈 추출 (QualIT 논문 방식, MCPanal 이식).

시드 무관 단계 — temp 0.2 고정, (모델, 문서 해시, 프롬프트 버전)으로 캐시되어
N시드 반복실험에서 재사용된다.
"""
from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from config import LLM_WORKERS, PROMPT_VERSION
from core.llm.ollama_client import OllamaClient
from core.utils.hashing import text_hash
from core.utils.progress import ProgressCallback

logger = logging.getLogger(__name__)

# 연속 연결 실패가 이 횟수에 도달하면 남은 문서를 포기하고 즉시 중단한다.
# (Ollama가 죽었는데 수천 건을 각각 재시도하며 시간을 낭비하는 것을 방지)
_CIRCUIT_BREAKER_THRESHOLD = 8


class OllamaUnavailableError(RuntimeError):
    pass


def determine_keyphrase_count(text: str) -> int:
    """문서 길이에 따라 키프레이즈 수 자동 결정 (MCPanal 기준)."""
    n = len(text.strip())
    if n < 30:
        return 2
    if n <= 300:
        return 3
    if n <= 800:
        return 5
    if n <= 2500:
        return 8
    return 12


def _build_prompt(text: str, kp_count: int) -> str:
    return f"""다음 문서에서 핵심 키프레이즈를 {kp_count}개 이하로 추출해주세요.

문서:
{text}

요구사항:
- 핵심 개념이나 주제를 나타내는 구나 단어
- 각 키프레이즈는 2-10단어 정도
- 쉼표로 구분하여 나열
- 예시: "인공지능 정책", "돌봄 서비스", "저출산 대응"

키프레이즈:"""


def _parse_keyphrases(raw: str, kp_count: int) -> List[str]:
    return [
        kp.strip().strip('"\'')
        for kp in re.split(r"[,，\n]", raw)
        if kp.strip() and len(kp.strip()) >= 2
    ][:kp_count]


def extract_keyphrases(
    texts: List[str],
    llm: OllamaClient,
    model: str,
    max_per_doc: Optional[int] = None,
    workers: int = LLM_WORKERS,
    candidate_indices: Optional[List[int]] = None,
    progress: Optional[ProgressCallback] = None,
) -> List[List[str]]:
    """문서별 키프레이즈 목록. candidate_indices가 주어지면 그 문서만 LLM 호출
    (나머지는 빈 리스트 — 문서 할당은 최근접 센트로이드로 처리됨).

    Ollama가 응답하지 않으면(연속 연결 실패) OllamaUnavailableError로 즉시 중단한다.
    """
    if llm.ping() is None:
        raise OllamaUnavailableError(
            "Ollama에 연결할 수 없습니다. `ollama serve`(또는 Ollama 앱)를 확인하세요."
        )

    consecutive_failures = 0
    failure_lock = threading.Lock()
    aborted = threading.Event()

    def _one(idx: int, text: str) -> tuple[int, List[str]]:
        nonlocal consecutive_failures
        if aborted.is_set():
            return idx, []
        stripped = (text or "").strip()
        if not stripped:
            return idx, []
        kp_count = max_per_doc if max_per_doc is not None else determine_keyphrase_count(stripped)
        try:
            raw = llm.chat_text(
                model,
                [{"role": "user", "content": _build_prompt(stripped, kp_count)}],
                temperature=0.2,
                max_tokens=200,
                num_ctx=16384,
                cache_tag=f"kp:{PROMPT_VERSION}:{text_hash(stripped)}",
            )
            with failure_lock:
                consecutive_failures = 0
            return idx, _parse_keyphrases(raw, kp_count)
        except Exception as e:
            with failure_lock:
                consecutive_failures += 1
                if consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                    aborted.set()
            logger.warning("키프레이즈 추출 실패 (문서 %d): %s", idx, e)
            return idx, []

    targets = (
        [(i, texts[i]) for i in candidate_indices]
        if candidate_indices is not None
        else list(enumerate(texts))
    )
    results: List[List[str]] = [[] for _ in texts]
    n = len(targets)
    if n == 0:
        return results
    max_workers = max(1, min(workers, n))
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_one, i, t) for i, t in targets]
        for future in as_completed(futures):
            idx, kps = future.result()
            results[idx] = kps
            done += 1
            if progress and (done % 10 == 0 or done == n):
                progress(done / n, f"키프레이즈 추출 {done}/{n}")
    if aborted.is_set():
        raise OllamaUnavailableError(
            f"Ollama 연속 {_CIRCUIT_BREAKER_THRESHOLD}회 호출 실패로 중단했습니다. "
            "서버 상태 확인 후 재실행하세요 (성공분은 캐시되어 재사용됩니다)."
        )
    return results
