"""TopicModel 공통 계약 + 레지스트리.

공정성 규칙: fit()은 공유 bge-m3 임베딩 행렬과 공유 Kiwi 토큰(corpus.tokenized)을
받는다. 모델이 평가용 임베딩/토큰을 자체 계산하지 않는다.
"""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

import numpy as np

from core.types import Corpus, TopicResult
from core.utils.hashing import stable_hash
from core.utils.progress import ProgressCallback


@dataclass(frozen=True)
class ModelConfig:
    name: str = ""                       # 레지스트리 키 (bertopic/lda/nmf/qualit)
    display_name: str = ""               # UI 표시명 (프리셋 구분용)
    n_topics: Optional[int] = None       # None = 자동 결정 (HDBSCAN/silhouette)

    def hash(self) -> str:
        return stable_hash(self)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["__config_class__"] = type(self).__name__
        return d

    @property
    def label(self) -> str:
        return self.display_name or self.name


class TopicModel(ABC):
    requires_embeddings: bool = True
    requires_llm: bool = False

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def fit(
        self,
        corpus: Corpus,
        embeddings: Optional[np.ndarray],
        seed: int,
        progress: Optional[ProgressCallback] = None,
    ) -> TopicResult:
        ...


# name → (모델 클래스, 설정 클래스)
MODEL_REGISTRY: Dict[str, tuple[Type[TopicModel], Type[ModelConfig]]] = {}


def register(name: str, config_cls: Type[ModelConfig]):
    def _wrap(model_cls: Type[TopicModel]):
        MODEL_REGISTRY[name] = (model_cls, config_cls)
        return model_cls

    return _wrap


def create_model(config: ModelConfig, **kwargs: Any) -> TopicModel:
    if config.name not in MODEL_REGISTRY:
        raise KeyError(f"등록되지 않은 모델: {config.name} (등록: {list(MODEL_REGISTRY)})")
    model_cls, _ = MODEL_REGISTRY[config.name]
    return model_cls(config, **kwargs)


def config_from_dict(data: Dict[str, Any]) -> ModelConfig:
    """RunRecord에 직렬화된 설정을 복원."""
    data = dict(data)
    data.pop("__config_class__", None)
    name = data.get("name", "")
    if name not in MODEL_REGISTRY:
        raise KeyError(f"등록되지 않은 모델: {name}")
    _, config_cls = MODEL_REGISTRY[name]
    valid_fields = {f.name for f in dataclasses.fields(config_cls)}
    return config_cls(**{k: v for k, v in data.items() if k in valid_fields})
