"""LLM-as-judge 질적 평가 (MCPanal 루브릭 이식): 명확성·일관성·구분성·근거성.

(토픽 집합 해시, judge 모델, 루브릭 버전)으로 캐시되어 반복 호출 비용이 없다.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from config import DEFAULT_JUDGE_MODEL
from core.llm.json_utils import parse_json_response
from core.llm.ollama_client import OllamaClient, get_default_client
from core.types import Corpus, TopicResult
from core.utils.hashing import stable_hash

logger = logging.getLogger(__name__)

RUBRIC_VERSION = "v1"

DIMENSIONS = [
    ("clarity", "명확성", "토픽 라벨 자체가 즉시 이해되는가"),
    ("consistency", "일관성", "대표 문서 묶음이 내부적으로 통일된 주제를 말하는가"),
    ("distinctiveness", "구분성", "다른 토픽과 경계가 분명한가"),
    ("groundedness", "근거성", "라벨·키워드가 대표 문서로 설명되는가"),
]


def _build_messages(algorithm: str, topics: List[Dict]) -> List[Dict[str, str]]:
    all_labels = [str(t.get("label") or "") for t in topics]
    blocks = []
    for t in topics:
        reps = "\n".join(f"- {r}" for r in (t.get("representatives") or [])) or "- 대표 문서 없음"
        kws = ", ".join(t.get("keywords") or []) or "(없음)"
        blocks.append(
            f"""[토픽 {t['topic_id']}]
라벨: {t.get('label') or ''}
키워드: {kws}
대표 문서:
{reps}"""
        )
    rubric = "\n".join(f"- {name}: {desc}" for _, name, desc in DIMENSIONS)
    user_prompt = f"""다음은 토픽모델링 결과에 대한 질적 품질 평가 과제입니다.

알고리즘: {algorithm}
전체 토픽 라벨: {', '.join(all_labels)}

평가축(각 항목 1~5점):
{rubric}

평가 원칙:
1. 명확성은 이름 자체가 이해되는지를 본다.
2. 일관성은 대표 문서 묶음의 내부 통일성을 본다.
3. 구분성은 다른 토픽과의 경계를 본다.
4. 근거성은 토픽명이 대표 문서로 설명되는지를 본다.
5. 반드시 JSON 배열만 출력한다.

출력 형식:
[
  {{"topic_id": 0, "clarity_score": 4, "consistency_score": 3, "distinctiveness_score": 2, "groundedness_score": 4, "comment": "한 줄 총평"}}
]

평가 대상 토픽:
{chr(10).join(blocks)}
"""
    return [
        {
            "role": "system",
            "content": "당신은 토픽모델링 결과를 연구방법론 관점에서 질적으로 평가하는 전문가다. 반드시 JSON만 출력한다.",
        },
        {"role": "user", "content": user_prompt},
    ]


def judge_topics(
    result: TopicResult,
    corpus: Corpus,
    llm: Optional[OllamaClient] = None,
    model: str = DEFAULT_JUDGE_MODEL,
    n_representatives: int = 3,
) -> Dict[str, float]:
    """토픽 집합 전체를 1회 호출로 평가, 차원별 평균 점수를 반환 (1~5)."""
    llm = llm or get_default_client()
    topics = []
    for tid in result.topic_ids:
        member_idx = np.where(result.assignments == tid)[0][:n_representatives]
        topics.append(
            {
                "topic_id": int(tid),
                "label": result.topic_labels.get(tid, ""),
                "keywords": (result.topic_keywords.get(tid) or [])[:8],
                "representatives": [corpus.docs[i].text[:200] for i in member_idx],
            }
        )
    if not topics:
        return {}

    messages = _build_messages(result.model_name, topics)
    cache_tag = "judge:" + stable_hash(
        {"rubric": RUBRIC_VERSION, "model": model, "topics": topics}, length=32
    )
    try:
        raw = llm.chat_text(
            model, messages, temperature=0.1, max_tokens=2500, cache_tag=cache_tag
        )
        evaluations = parse_json_response(raw)
    except Exception as e:
        logger.warning("LLM judge 실패: %s", e)
        return {}

    scores: Dict[str, List[float]] = {key: [] for key, _, _ in DIMENSIONS}
    for item in evaluations if isinstance(evaluations, list) else []:
        for key, _, _ in DIMENSIONS:
            try:
                value = float(item.get(f"{key}_score") or 0)
                if 1 <= value <= 5:
                    scores[key].append(value)
            except Exception:
                continue
    out: Dict[str, float] = {}
    for key, _, _ in DIMENSIONS:
        if scores[key]:
            out[f"judge_{key}"] = float(np.mean(scores[key]))
    if out:
        out["judge_overall"] = float(np.mean(list(out.values())))
    return out
