"""페이지 공용 헬퍼: 코퍼스 선택, Ollama 상태 표시, 진행 콜백."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import streamlit as st

from core.data.corpus_store import list_corpora, load_corpus
from core.llm.ollama_client import get_default_client
from core.types import Corpus


def render_ollama_status() -> None:
    with st.sidebar:
        client = get_default_client()
        host = client.ping()
        if host:
            st.success(f"Ollama 연결됨: {host}", icon="🟢")
        else:
            st.error("Ollama에 연결할 수 없습니다. `ollama serve` 확인", icon="🔴")


def select_corpus(key: str = "corpus_select") -> Optional[dict]:
    """저장된 코퍼스 선택 UI. 선택된 meta dict 반환."""
    corpora = list_corpora()
    if not corpora:
        st.warning("저장된 코퍼스가 없습니다. **1. 데이터 준비** 페이지에서 먼저 데이터를 준비하세요.")
        return None
    options = {
        f"{m['name']} · {m['regime']} · {m['n_docs']}건 · {m['content_hash'][:8]}": m
        for m in corpora
    }
    label = st.selectbox("분석 코퍼스", list(options.keys()), key=key)
    return options[label]


@st.cache_resource(show_spinner="코퍼스 로드 중...")
def cached_load_corpus(content_hash: str) -> Corpus:
    return load_corpus(content_hash)


@st.cache_resource(show_spinner="임베딩 로드 중 (캐시 활용)...")
def cached_embeddings(content_hash: str) -> np.ndarray:
    from core.data.prepare import get_corpus_embeddings

    corpus = cached_load_corpus(content_hash)
    return get_corpus_embeddings(corpus)


def load_selected(meta: dict) -> Tuple[Corpus, np.ndarray]:
    corpus = cached_load_corpus(meta["content_hash"])
    embeddings = cached_embeddings(meta["content_hash"])
    return corpus, embeddings


def st_progress_callback(placeholder):
    bar = placeholder.progress(0.0, text="준비 중")

    def _cb(fraction: float, message: str) -> None:
        bar.progress(max(0.0, min(1.0, fraction)), text=message)

    return _cb
