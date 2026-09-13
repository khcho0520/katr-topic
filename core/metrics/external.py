"""외적 타당도 지표: 정답 레이블(기사 카테고리) 대비 토픽 할당 일치도.

BigKinds/MCPanal 코퍼스의 카테고리 메타(예: "사회>사건_사고 | 사회>여성")에서
대분류("사회")를 정답 레이블로 삼아 NMI, ARI, Purity를 계산한다.
할당된(-1 제외) + 레이블 있는 문서만 대상으로 하며, 그 비율(label_coverage)을 함께 기록.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from core.types import Corpus

# 코퍼스의 절반 이상(최소 10건)에 레이블이 있어야 외적 평가를 수행한다
_MIN_LABEL_FRACTION = 0.5
_MIN_LABELED_DOCS = 10


def extract_major_category(raw: str) -> str:
    """"사회>사건_사고 | 사회>여성" → "사회" (첫 항목의 대분류)."""
    first = (raw or "").split("|")[0].strip()
    return first.split(">")[0].strip()


def extract_labels(corpus: Corpus) -> Optional[List[Optional[str]]]:
    """문서별 대분류 레이블. 레이블 비율이 기준 미달이면 None (외적 평가 생략)."""
    labels: List[Optional[str]] = []
    n_present = 0
    for doc in corpus.docs:
        major = extract_major_category(str(doc.meta.get("category") or ""))
        labels.append(major or None)
        if major:
            n_present += 1
    if n_present < max(_MIN_LABELED_DOCS, int(len(corpus.docs) * _MIN_LABEL_FRACTION)):
        return None
    return labels


def _purity(topic_ids: np.ndarray, label_ids: np.ndarray) -> float:
    total = len(topic_ids)
    if total == 0:
        return 0.0
    correct = 0
    for tid in np.unique(topic_ids):
        member_labels = label_ids[topic_ids == tid]
        correct += Counter(member_labels.tolist()).most_common(1)[0][1]
    return correct / total


def compute_external_metrics(
    assignments: np.ndarray,
    labels: List[Optional[str]],
) -> Dict[str, float]:
    """할당·레이블 모두 있는 문서에 대한 NMI/ARI/Purity."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    mask = np.array(
        [a >= 0 and bool(l) for a, l in zip(assignments, labels)], dtype=bool
    )
    n = int(mask.sum())
    if n < _MIN_LABELED_DOCS:
        return {}
    topic_ids = np.asarray(assignments)[mask]
    label_names = [l for a, l in zip(assignments, labels) if a >= 0 and l]
    label_map = {name: i for i, name in enumerate(sorted(set(label_names)))}
    label_ids = np.array([label_map[l] for l in label_names])
    if len(label_map) < 2 or len(np.unique(topic_ids)) < 2:
        return {}
    return {
        "nmi": float(normalized_mutual_info_score(label_ids, topic_ids)),
        "ari": float(adjusted_rand_score(label_ids, topic_ids)),
        "purity": _purity(topic_ids, label_ids),
        "label_coverage": n / len(assignments),
        "n_label_classes": float(len(label_map)),
    }
