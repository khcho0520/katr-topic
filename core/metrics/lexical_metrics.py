"""어휘 기반 지표 (MCPanal 이식): NPMI, Topic Diversity.

공정성 규칙: 모든 모델의 키워드를 Kiwi 명사 토큰으로 정규화한 뒤 계산한다.
LLM이 생성한 키프레이즈도 동일하게 정규화되어 표면형 불일치의 영향을 받지 않는다.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Set

import numpy as np

from core.data.tokenize_ko import normalize_keyword


def normalize_topic_keywords(topic_keywords: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """키워드(구 단위 포함)를 Kiwi 토큰으로 펼치고 토픽 내 중복 제거."""
    normalized: Dict[int, List[str]] = {}
    for tid, kws in topic_keywords.items():
        seen: Set[str] = set()
        tokens: List[str] = []
        for kw in kws:
            for tok in normalize_keyword(kw):
                if tok not in seen:
                    seen.add(tok)
                    tokens.append(tok)
        normalized[tid] = tokens
    return normalized


def topic_diversity(topic_keywords: Dict[int, List[str]], top_k: int = 10) -> float:
    """고유 키워드 비율: unique(top-k 키워드) / (토픽 수 × top_k)."""
    if not topic_keywords:
        return 0.0
    selected: List[str] = []
    for kws in topic_keywords.values():
        selected.extend(kws[:top_k])
    if not selected:
        return 0.0
    denom = len(topic_keywords) * top_k
    return float(len(set(selected)) / denom) if denom else 0.0


def npmi_for_topic(
    topic_keywords: List[str],
    baseline_token_sets: List[Set[str]],
    max_vocab: int = 10,
    min_df: int = 2,
) -> float:
    """한 토픽의 키워드 쌍 NPMI (기준 말뭉치 = 코퍼스 전체 문서 토큰 집합)."""
    candidate = list(dict.fromkeys(topic_keywords))[:max_vocab]
    candidate_set = set(candidate)
    if len(candidate_set) < 2 or len(baseline_token_sets) < 2:
        return 0.0

    N = len(baseline_token_sets)
    df: Counter = Counter()
    pair_df: Counter = Counter()
    for doc_tokens in baseline_token_sets:
        inter = sorted(candidate_set.intersection(doc_tokens))
        for t in inter:
            df[t] += 1
        for i in range(len(inter)):
            for j in range(i + 1, len(inter)):
                pair_df[(inter[i], inter[j])] += 1

    values: List[float] = []
    for (w1, w2), c12 in pair_df.items():
        c1, c2 = df.get(w1, 0), df.get(w2, 0)
        if c12 < min_df or c1 < min_df or c2 < min_df:
            continue
        p12, p1, p2 = c12 / N, c1 / N, c2 / N
        denom = -math.log(p12)
        if denom == 0:
            continue
        values.append(math.log(p12 / (p1 * p2)) / denom)
    return float(np.mean(values)) if values else 0.0


def compute_lexical_metrics(
    topic_keywords: Dict[int, List[str]],
    corpus_tokenized: List[List[str]],
) -> Dict[str, float]:
    normalized = normalize_topic_keywords(topic_keywords)
    baseline_sets = [set(toks) for toks in corpus_tokenized]
    npmi_values = [
        npmi_for_topic(kws, baseline_sets) for kws in normalized.values() if kws
    ]
    return {
        "npmi": float(np.mean(npmi_values)) if npmi_values else 0.0,
        "diversity": topic_diversity(normalized),
    }
