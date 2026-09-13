"""클러스터링 대상 표현(representation) 구성 — ablation 지점.

- keyphrase: 키프레이즈 임베딩을 클러스터링 (QualIT 논문)
- doc: 문서 임베딩을 직접 클러스터링 (BERTopic과 유사한 대조군)
- joint: alpha·문서임베딩 + (1-alpha)·키프레이즈임베딩 (개선 후보 —
  키프레이즈의 주제 신호와 문서 맥락을 결합)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class RepresentationOutput:
    vectors: np.ndarray            # 클러스터링 대상 행렬
    item_doc_idx: List[int]        # 각 항목 → 원본 문서 인덱스
    item_keyphrases: List[str]     # 각 항목의 키프레이즈 텍스트 (doc 모드는 "")


def build_representation(
    mode: str,
    keyphrases_per_doc: List[List[str]],
    doc_embeddings: np.ndarray,
    kp_embedding_lookup: Dict[str, np.ndarray],
    joint_alpha: float = 0.5,
) -> RepresentationOutput:
    if mode == "doc":
        return RepresentationOutput(
            vectors=np.asarray(doc_embeddings, dtype=np.float32),
            item_doc_idx=list(range(len(doc_embeddings))),
            item_keyphrases=[""] * len(doc_embeddings),
        )

    vectors: List[np.ndarray] = []
    item_doc_idx: List[int] = []
    item_keyphrases: List[str] = []
    for doc_idx, kps in enumerate(keyphrases_per_doc):
        for kp in kps:
            kp_emb = kp_embedding_lookup.get(kp)
            if kp_emb is None:
                continue
            if mode == "keyphrase":
                vec = kp_emb
            elif mode == "joint":
                doc_emb = doc_embeddings[doc_idx]
                vec = joint_alpha * doc_emb + (1.0 - joint_alpha) * kp_emb
            else:
                raise ValueError(f"알 수 없는 표현 모드: {mode}")
            vectors.append(np.asarray(vec, dtype=np.float32))
            item_doc_idx.append(doc_idx)
            item_keyphrases.append(kp)

    if not vectors:
        raise ValueError("클러스터링할 키프레이즈 임베딩이 없습니다.")
    return RepresentationOutput(
        vectors=np.stack(vectors),
        item_doc_idx=item_doc_idx,
        item_keyphrases=item_keyphrases,
    )
