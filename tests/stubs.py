"""테스트용 스텁: Ollama 없이 QualIT 파이프라인 계약을 검증한다."""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

import numpy as np


class StubLLM:
    """키프레이즈 프롬프트에는 문서 앞 단어 2개, 테마 프롬프트에는 고정 라벨을 반환."""

    def __init__(self):
        self.call_count = 0

    def ping(self):
        return "stub"

    def chat_text(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        self.call_count += 1
        prompt = messages[-1]["content"]
        if "키프레이즈를" in prompt and "문서:" in prompt:
            match = re.search(r"문서:\n(.*?)\n\n요구사항", prompt, re.DOTALL)
            text = match.group(1).strip() if match else ""
            words = [w for w in text.split() if len(w) >= 2][:2]
            return ", ".join(words) if words else "일반 주제"
        if "주제(테마)로 요약" in prompt:
            match = re.search(r"키프레이즈: ([^\n]+)", prompt)
            first = (match.group(1).split(",")[0].strip() if match else "테마")
            return f"{first} 관련 주제"
        if "근거하는지 판정" in prompt:
            return "[0, 1]"
        return "[]"


class StubEmbedder:
    """문자 3-gram 해싱 임베딩 — 결정적이며, 겹치는 텍스트는 유사한 벡터를 얻는다."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.call_count = 0

    def embed_texts(self, texts: List[str], progress=None) -> np.ndarray:
        self.call_count += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            padded = f"  {text}  "
            for j in range(len(padded) - 2):
                gram = padded[j : j + 3]
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


def synthetic_corpus(n_per_theme: int = 15):
    """두 주제(AI 정책 / 부동산)로 뚜렷이 갈리는 합성 한국어 코퍼스."""
    from core.data.tokenize_ko import tokenize_corpus
    from core.types import Corpus, Document
    from core.utils.hashing import texts_hash

    ai = [
        f"정부 인공지능 정책 발표 {i}차 규제 개편과 산업 지원 방안" for i in range(n_per_theme)
    ]
    estate = [
        f"서울 아파트 가격 상승 {i}주차 부동산 시장 전세 동향" for i in range(n_per_theme)
    ]
    texts = ai + estate
    docs = [
        Document(doc_id=f"d{i}", text=t, source_id="synthetic") for i, t in enumerate(texts)
    ]
    corpus = Corpus(
        regime="headlines",
        docs=docs,
        content_hash=texts_hash(texts),
        name="synthetic",
        tokenized=tokenize_corpus(texts),
    )
    return corpus


def blob_embeddings(n_per_cluster: int, dim: int = 32, n_clusters: int = 2, spread: float = 0.05):
    """잘 분리된 가우시안 블롭 임베딩 (결정적)."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(n_clusters, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rows = []
    for c in range(n_clusters):
        rows.append(centers[c] + rng.normal(scale=spread, size=(n_per_cluster, dim)))
    matrix = np.vstack(rows).astype(np.float32)
    labels = np.repeat(np.arange(n_clusters), n_per_cluster)
    return matrix, labels
