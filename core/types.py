"""공용 데이터 타입: 코퍼스, 토픽 결과, 실험 기록."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str                 # 분석 단위 텍스트 (헤드라인 1건 또는 논문 청크 1개)
    source_id: str            # 출처 (파일명/행 번호 등)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Corpus:
    regime: Literal["headlines", "papers"]
    docs: List[Document]
    content_hash: str                                  # 캐시 키 루트
    name: str = ""
    tokenized: Optional[List[List[str]]] = None        # Kiwi 명사 토큰 (1회 계산 후 공유)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def texts(self) -> List[str]:
        return [d.text for d in self.docs]

    def __len__(self) -> int:
        return len(self.docs)


@dataclass
class TopicResult:
    model_name: str
    config_hash: str
    seed: int
    assignments: np.ndarray                    # len == n_docs; -1 = 아웃라이어/미할당
    topic_keywords: Dict[int, List[str]]       # topic_id → 상위 키워드
    topic_labels: Dict[int, str]               # topic_id → 사람이 읽는 라벨
    doc_topic_dist: Optional[np.ndarray] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    runtime_sec: float = 0.0

    @property
    def topic_ids(self) -> List[int]:
        return sorted(t for t in set(int(a) for a in self.assignments) if t >= 0)

    def topic_sizes(self) -> Dict[int, int]:
        return {t: int(np.sum(self.assignments == t)) for t in self.topic_ids}


@dataclass
class MetricReport:
    scalars: Dict[str, float]
    per_topic: Optional[Any] = None            # pd.DataFrame


@dataclass
class RunRecord:
    run_id: str                                # f"{model}_{config_hash}_s{seed}"
    experiment_id: str
    corpus_hash: str
    corpus_name: str
    regime: str
    model_name: str
    config: Dict[str, Any]
    config_hash: str
    seed: int
    metrics: Dict[str, float]
    topics: Dict[str, Any]                     # {topic_id: {label, keywords, size}}
    created_at: str
    status: Literal["done", "failed"] = "done"
    error: str = ""
    # 시드 간 안정성(ARI) 계산용 원자료. 과거 기록에는 없을 수 있음(빈 리스트).
    assignments: List[int] = field(default_factory=list)
