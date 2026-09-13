"""대표 문서 선택(층화 샘플링) + 키프레이즈 서킷브레이커 검증."""
from __future__ import annotations

import numpy as np
import pytest

from core.models.qualit.candidates import select_candidate_indices
from core.models.qualit.keyphrase import OllamaUnavailableError, extract_keyphrases
from tests.stubs import blob_embeddings


def test_small_corpus_returns_all():
    matrix, _ = blob_embeddings(10, dim=16, n_clusters=2)
    assert select_candidate_indices(matrix, max_docs=500) == list(range(20))
    assert select_candidate_indices(matrix, max_docs=0) == list(range(20))


def test_cap_respected_and_deterministic():
    matrix, labels = blob_embeddings(300, dim=16, n_clusters=3)
    picked1 = select_candidate_indices(matrix, max_docs=60)
    picked2 = select_candidate_indices(matrix, max_docs=60)
    assert picked1 == picked2                       # 결정적 (시드 간 캐시 재사용 전제)
    assert len(picked1) == 60
    assert len(set(picked1)) == 60
    # 층화: 세 블롭 모두에서 문서가 뽑혀야 함
    picked_labels = {int(labels[i]) for i in picked1}
    assert picked_labels == {0, 1, 2}


class _DeadLLM:
    """ping은 되지만 chat은 전부 연결 실패하는 스텁 (Ollama 중도 다운 시나리오)."""

    def ping(self):
        return "stub"

    def chat_text(self, *args, **kwargs):
        raise ConnectionError("Connection refused")


class _NoPingLLM:
    def ping(self):
        return None

    def chat_text(self, *args, **kwargs):
        raise AssertionError("ping 실패 시 호출되면 안 됨")


def test_circuit_breaker_aborts_fast():
    texts = [f"문서 {i} 내용" for i in range(100)]
    with pytest.raises(OllamaUnavailableError):
        extract_keyphrases(texts, _DeadLLM(), "qwen3.5:9b", workers=4)


def test_ping_failure_fails_immediately():
    with pytest.raises(OllamaUnavailableError):
        extract_keyphrases(["문서"], _NoPingLLM(), "qwen3.5:9b")


def test_candidate_indices_limits_llm_calls():
    calls = []

    class _CountingLLM:
        def ping(self):
            return "stub"

        def chat_text(self, model, messages, **kwargs):
            calls.append(1)
            return "키워드 하나, 키워드 둘"

    texts = [f"문서 {i} 내용입니다" for i in range(50)]
    results = extract_keyphrases(
        texts, _CountingLLM(), "m", candidate_indices=[0, 10, 20], workers=2
    )
    assert len(calls) == 3
    assert results[0] and results[10] and results[20]
    assert results[1] == [] and results[49] == []
