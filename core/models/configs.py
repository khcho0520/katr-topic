"""프리셋 설정: 베이스라인 + QualIT 논문 재현 + 개선 후보(ablation) 그리드.

논문 실험은 이 프리셋 목록에서 골라 seeds × configs 그리드로 실행한다.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

from core.models.base import ModelConfig
from core.models.bertopic_model import BERTopicConfig
from core.models.katr_model import KATRConfig
from core.models.lda_model import LDAConfig
from core.models.nmf_model import NMFConfig
from core.models.qualit.pipeline import QualITConfig

# 2026-07 3개월 코퍼스 K=20 실험(exp_bigkinds_2026-01_3개월_5e438796)에 근거해 선별.
# 제거된 변형(qualit_hdbscan/ctfidf/percentile/llmverify/refine/improved, katr_strong_anchor)은
# 성능 미달 또는 미검증 — 필요 시 페이지 2의 ablation 폼에서 수동 구성 가능.
PRESETS: Dict[str, ModelConfig] = {
    # ---- 제안 모델: KATR (Keyphrase-Anchored Topic Refinement) ----
    # 실측 최종형: α=1(대조군이 앵커보다 우세) + 코어 문서 c-TF-IDF 키워드.
    # coherence·purity에서 BERTopic-full과 동률, 안정성 ARI +0.16, 전수 커버리지.
    "katr_core": KATRConfig(
        display_name="KATR-core(코어키워드)",
        alpha=1.0,
        keyword_core_fraction=0.5,
    ),
    # ablation 짝: 앵커 기여분(katr vs katr_noanchor), 코어 키워드 기여분(katr_noanchor vs katr_core)
    "katr": KATRConfig(display_name="KATR (제안)"),
    "katr_noanchor": KATRConfig(
        display_name="KATR-α1(앵커무효 대조군)",
        alpha=1.0,
    ),
    # 앵커 강화 조건 (논문 표 1~3의 α=0.3 행)
    "katr_alpha03": KATRConfig(
        display_name="KATR-α0.3(앵커강화)",
        alpha=0.3,
    ),
    # ---- 레거시 비교군 ----
    "bertopic": BERTopicConfig(display_name="BERTopic"),
    "bertopic_full": BERTopicConfig(
        display_name="BERTopic-full(전수할당)",
        assign_outliers=True,
    ),
    "lda": LDAConfig(display_name="LDA"),
    "nmf": NMFConfig(display_name="NMF"),
    # ---- QualIT 참조 모델 ----
    "qualit_paper": QualITConfig(display_name="QualIT (논문 재현)"),
    # QualIT 계열 실측 최고 변형 (coherence 0.619) — QualIT 개선 한계 서술용
    "qualit_joint": QualITConfig(
        display_name="QualIT+결합표현(α=0.5)",
        representation="joint",
        joint_alpha=0.5,
    ),
}


def get_preset(key: str) -> ModelConfig:
    if key not in PRESETS:
        raise KeyError(f"알 수 없는 프리셋: {key} (가능: {list(PRESETS)})")
    return PRESETS[key]


def apply_common_conditions(
    configs: List[ModelConfig],
    n_topics: int = 0,
    bertopic_min_cluster_size: int = 0,
    qualit_max_llm_docs: Optional[int] = None,
) -> List[ModelConfig]:
    """공정 비교용 조건 통일: 선택한 설정들에 공통 조건을 일괄 적용한다.

    - n_topics > 0: 모든 모델에 동일 K 고정 (K 교란 제거).
      · LDA/NMF: n_components=K
      · BERTopic: nr_topics=K로 축소 — HDBSCAN이 K개 이상 찾도록
        min_cluster_size도 함께 낮추는 것을 권장
      · QualIT: 클러스터링 target K=K (단, clustering="hdbscan" 변형은 K-free라 미적용)
    - bertopic_min_cluster_size > 0: BERTopic HDBSCAN 최소 클러스터 크기 고정
      (기본 n/20은 대량 코퍼스에서 과대해져 토픽 수가 붕괴함)
    - qualit_max_llm_docs is not None: QualIT 계열 LLM 대표 문서 상한 통일

    조건이 적용된 설정은 display_name에 접미사가 붙어 결과 표에서 구분된다.
    """
    out: List[ModelConfig] = []
    for config in configs:
        c = config
        suffix: List[str] = []
        if n_topics > 0:
            if isinstance(c, QualITConfig):
                c = dataclasses.replace(
                    c, n_topics=n_topics, max_topics=max(c.max_topics, n_topics)
                )
            else:
                c = dataclasses.replace(c, n_topics=n_topics)
            suffix.append(f"K={n_topics}")
        if bertopic_min_cluster_size > 0 and isinstance(c, BERTopicConfig):
            c = dataclasses.replace(c, min_cluster_size=bertopic_min_cluster_size)
            suffix.append(f"mcs={bertopic_min_cluster_size}")
        if qualit_max_llm_docs is not None:
            if isinstance(c, QualITConfig) and c.max_llm_docs != qualit_max_llm_docs:
                c = dataclasses.replace(c, max_llm_docs=qualit_max_llm_docs)
                suffix.append(f"llm≤{qualit_max_llm_docs}")
            elif isinstance(c, KATRConfig) and c.anchor_pool_size != qualit_max_llm_docs:
                # KATR의 LLM 예산은 앵커 후보 풀 크기 — 동일 조건으로 통일
                c = dataclasses.replace(c, anchor_pool_size=qualit_max_llm_docs)
                suffix.append(f"llm≤{qualit_max_llm_docs}")
        if suffix:
            c = dataclasses.replace(
                c, display_name=f"{config.label} ({', '.join(suffix)})"
            )
        out.append(c)
    return out


def baseline_keys() -> List[str]:
    return ["bertopic", "lda", "nmf"]


def qualit_keys() -> List[str]:
    return [k for k in PRESETS if k.startswith("qualit")]
