"""학술 논문 텍스트 정제 (MCPanal _clean_document_text 계열 이식).

참고문헌 이후 절단, 표/그림 캡션·페이지 번호·URL 등 잡음 제거.
"""
from __future__ import annotations

import re

# 참고문헌 섹션 시작 패턴 — 이후 내용은 잘라낸다
_REFERENCES_PATTERN = re.compile(
    r"^\s*(참\s*고\s*문\s*헌|참고자료|References?|REFERENCES?|Bibliography)\s*$",
    re.MULTILINE,
)

_NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*[<\[(]?\s*(표|그림|Table|Figure|Fig\.?)\s*\d+[>\])]?.{0,80}$", re.IGNORECASE),
    re.compile(r"^\s*-?\s*\d+\s*-?\s*$"),                      # 페이지 번호
    re.compile(r"^\s*(doi|DOI)\s*[:：]?\s*\S+\s*$"),
    re.compile(r"^\s*https?://\S+\s*$"),
    re.compile(r"^\s*(Vol\.?|No\.?|pp\.?)\s*[\d.,\s-]+\s*$", re.IGNORECASE),
    re.compile(r"^\s*[·•*─—=_-]{3,}\s*$"),                     # 구분선
]


def clean_document_text(text: str) -> str:
    if not (text or "").strip():
        return ""

    # 참고문헌 이후 절단
    match = _REFERENCES_PATTERN.search(text)
    if match and match.start() > len(text) * 0.3:
        text = text[: match.start()]

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if any(p.match(stripped) for p in _NOISE_LINE_PATTERNS):
            continue
        # 인라인 URL/이메일 제거
        stripped = re.sub(r"https?://\S+", " ", stripped)
        stripped = re.sub(r"\S+@\S+\.\S+", " ", stripped)
        lines.append(stripped)

    cleaned = "\n".join(lines)
    # 하이픈 줄바꿈 결합, 과도한 공백 정리
    cleaned = re.sub(r"-\n(?=[a-z가-힣])", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
