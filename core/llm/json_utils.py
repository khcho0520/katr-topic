"""LLM 응답에서 JSON 블록 추출/파싱 (MCPanal _extract_json_block 이식)."""
from __future__ import annotations

import json
import re
from typing import Any, Optional


def extract_json_block(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None
    for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            return match.group(0)
    return None


def parse_json_response(raw_text: str) -> Any:
    """LLM 응답에서 첫 JSON 배열/객체를 파싱. 실패 시 ValueError."""
    block = extract_json_block(raw_text)
    if not block:
        raise ValueError("응답에서 JSON을 찾지 못했습니다.")
    return json.loads(block)
