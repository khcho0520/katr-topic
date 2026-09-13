"""Kiwi 형태소 분석 기반 한국어 토크나이저.

모든 모델(LDA/NMF 벡터라이저, BERTopic c-TF-IDF)과 어휘 지표(NPMI, diversity)가
이 토크나이저를 공유한다 — MCPanal의 정규식 토크나이저 약점을 수정한 공정성 규칙.
"""
from __future__ import annotations

import re
import threading
from functools import lru_cache
from typing import List, Optional, Set

from config import STOPWORDS_PATH

# 명사류 + 외국어(영문 약어 등)만 사용
_ALLOWED_TAGS = {"NNG", "NNP", "SL"}
_MIN_TOKEN_LEN = 2

_kiwi = None
_kiwi_lock = threading.Lock()
_stopwords: Optional[Set[str]] = None


def _get_kiwi():
    global _kiwi
    with _kiwi_lock:
        if _kiwi is None:
            from kiwipiepy import Kiwi

            _kiwi = Kiwi()
        return _kiwi


def get_stopwords() -> Set[str]:
    global _stopwords
    if _stopwords is None:
        words: Set[str] = set()
        if STOPWORDS_PATH.exists():
            for line in STOPWORDS_PATH.read_text(encoding="utf-8").splitlines():
                w = line.strip()
                if w and not w.startswith("#"):
                    words.add(w)
        _stopwords = words
    return _stopwords


def tokenize(text: str) -> List[str]:
    """텍스트 → Kiwi 명사/외국어 토큰 리스트 (불용어·1글자 제거, 영문은 소문자화)."""
    if not (text or "").strip():
        return []
    kiwi = _get_kiwi()
    stopwords = get_stopwords()
    tokens: List[str] = []
    # Kiwi.analyze는 스레드 안전하지 않을 수 있으므로 락으로 직렬화
    with _kiwi_lock:
        analyzed = kiwi.tokenize(text)
    for tok in analyzed:
        if tok.tag not in _ALLOWED_TAGS:
            continue
        form = tok.form.lower() if tok.tag == "SL" else tok.form
        if len(form) < _MIN_TOKEN_LEN:
            continue
        if form in stopwords:
            continue
        if re.fullmatch(r"[\d\W_]+", form):
            continue
        tokens.append(form)
    return tokens


def tokenize_corpus(texts: List[str]) -> List[List[str]]:
    return [tokenize(t) for t in texts]


@lru_cache(maxsize=65536)
def _tokenize_cached(text: str) -> tuple:
    return tuple(tokenize(text))


def normalize_keyword(keyword: str) -> List[str]:
    """LLM이 생성한 키워드/키프레이즈를 Kiwi 명사 토큰으로 정규화.

    어휘 지표(NPMI/diversity) 비교 시 LLM 계열 모델이 표면형 불일치로
    불이익/이득을 받지 않도록 모든 키워드를 동일 형태로 맞춘다.
    """
    return list(_tokenize_cached(keyword))
