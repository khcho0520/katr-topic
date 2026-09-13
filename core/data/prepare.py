"""코퍼스 준비 일괄 처리: Kiwi 토큰화 + bge-m3 임베딩 사전계산 + 디스크 저장.

UI 페이지와 헤드리스 스크립트가 공유하는 단일 진입점.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from core.data.corpus_store import load_embeddings, save_corpus, save_embeddings
from core.data.tokenize_ko import tokenize_corpus
from core.llm.embeddings import Embedder, get_default_embedder
from core.types import Corpus
from core.utils.progress import ProgressCallback, sub_progress


def prepare_corpus(
    corpus: Corpus,
    embedder: Optional[Embedder] = None,
    progress: Optional[ProgressCallback] = None,
    save: bool = True,
) -> Tuple[Corpus, np.ndarray]:
    """토큰화·임베딩을 채워서 (corpus, 임베딩 행렬)을 반환. 임베딩은 디스크 캐시 활용."""
    if corpus.tokenized is None:
        if progress:
            progress(0.0, f"Kiwi 토큰화 ({len(corpus)}건)")
        corpus.tokenized = tokenize_corpus(corpus.texts)
    if progress:
        progress(0.3, "임베딩 계산/캐시 로드")
    stored = load_embeddings(corpus.content_hash)
    if stored is not None and len(stored) == len(corpus):
        embeddings = stored
    else:
        embedder = embedder or get_default_embedder()
        embeddings = embedder.embed_texts(corpus.texts, progress=sub_progress(progress, 0.3, 0.95))
    if save:
        save_corpus(corpus)
        save_embeddings(corpus.content_hash, embeddings)
    if progress:
        progress(1.0, "코퍼스 준비 완료")
    return corpus, embeddings


def get_corpus_embeddings(
    corpus: Corpus, embedder: Optional[Embedder] = None
) -> np.ndarray:
    """저장된 코퍼스의 임베딩 행렬 로드 — npy 저장본 우선, 없으면 캐시/Ollama 계산."""
    stored = load_embeddings(corpus.content_hash)
    if stored is not None and len(stored) == len(corpus):
        return stored
    embedder = embedder or get_default_embedder()
    embeddings = embedder.embed_texts(corpus.texts)
    save_embeddings(corpus.content_hash, embeddings)
    return embeddings
