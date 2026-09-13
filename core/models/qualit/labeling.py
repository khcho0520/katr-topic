"""4단계: 토픽 라벨·키워드 생성 — ablation 지점.

- 라벨: LLM 테마 증류 (contrastive — 기존 라벨과 구분되게, MCPanal 이식).
  클러스터 멤버 해시로 캐시 → 시드가 달라도 동일 클러스터면 LLM 재호출 없음.
- 키워드: "llm" = 멤버 키프레이즈 상위 사용 (QualIT 논문) /
  "ctfidf" = 멤버 문서 Kiwi 토큰의 class-TF-IDF (BERTopic 방식 개선 후보)
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Dict, List, Optional

from core.llm.ollama_client import OllamaClient
from core.utils.hashing import text_hash

logger = logging.getLogger(__name__)

TOP_KEYWORDS = 10


def distill_theme(
    keyphrases: List[str],
    sample_texts: List[str],
    llm: OllamaClient,
    model: str,
    other_themes: Optional[List[str]] = None,
    max_length: int = 100,
) -> str:
    """클러스터 키프레이즈를 하나의 주제로 증류. 실패 시 키프레이즈 연결 폴백."""
    if not keyphrases:
        return ""
    keyphrases_str = ", ".join(keyphrases[:10])
    sample_text = sample_texts[0][:300] if sample_texts else ""
    distinct_instruction = ""
    if other_themes:
        others = "; ".join(other_themes[:8])
        distinct_instruction = f"""
아래 "이미 있는 다른 주제"와 의미가 겹치지 않도록, 이 클러스터만의 **고유한** 주제를 표현해주세요.
이미 있는 다른 주제: {others}
"""
    prompt = f"""다음 키프레이즈들을 종합하여 하나의 핵심 주제(테마)로 요약해주세요.

키프레이즈: {keyphrases_str}
{f'문서 샘플: {sample_text}' if sample_text else ''}
{distinct_instruction}

요구사항:
- 키프레이즈들의 공통 주제를 한 문장으로 간결하게 표현
- 50자 이내
- 예시: "돌봄 서비스와 가족 정책", "저출산 대응 및 출산 지원"

주제:"""
    try:
        # 캐시 키: 멤버 키프레이즈 집합 + contrastive 문맥 → 시드 간 재사용
        theme = llm.chat_text(
            model,
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150,
            cache_tag=f"theme:{text_hash(keyphrases_str)}:{text_hash(';'.join(other_themes or []))}",
        ).strip()
        for prefix in ("주제:", "주제 :", "Theme:", "답변:", "답변 :"):
            if theme.startswith(prefix):
                theme = theme[len(prefix):].strip()
        theme = theme.strip('"\'')
        if theme:
            return theme[:max_length]
    except Exception as e:
        logger.warning("주제 증류 실패: %s", e)
    return " · ".join(keyphrases[:3])


def keyphrase_keywords(member_keyphrases: List[str], top_k: int = TOP_KEYWORDS) -> List[str]:
    """멤버 키프레이즈 빈도 상위 top_k (QualIT 논문 방식)."""
    counter = Counter(member_keyphrases)
    return [kp for kp, _ in counter.most_common(top_k)]


def ctfidf_keywords(
    topic_token_docs: Dict[int, List[List[str]]],
    top_k: int = TOP_KEYWORDS,
) -> Dict[int, List[str]]:
    """토픽별 멤버 문서 Kiwi 토큰으로 class-TF-IDF 키워드 계산 (BERTopic 방식).

    tf(w, class) × log(1 + A / freq(w)) — A: 클래스당 평균 토큰 수.
    """
    class_counters: Dict[int, Counter] = {
        tid: Counter(t for doc in docs for t in doc) for tid, docs in topic_token_docs.items()
    }
    total_freq: Counter = Counter()
    for counter in class_counters.values():
        total_freq.update(counter)
    if not total_freq:
        return {tid: [] for tid in topic_token_docs}
    avg_class_size = sum(sum(c.values()) for c in class_counters.values()) / max(1, len(class_counters))

    keywords: Dict[int, List[str]] = {}
    for tid, counter in class_counters.items():
        class_total = sum(counter.values()) or 1
        scores = {
            w: (freq / class_total) * math.log(1.0 + avg_class_size / total_freq[w])
            for w, freq in counter.items()
        }
        keywords[tid] = [w for w, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
    return keywords
