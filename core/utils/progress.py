"""CLI/Streamlit 양쪽에서 쓸 수 있는 진행 콜백 프로토콜."""
from __future__ import annotations

from typing import Callable, Optional

# progress(fraction: float 0..1, message: str)
ProgressCallback = Callable[[float, str], None]


def null_progress(fraction: float, message: str) -> None:
    pass


def make_console_progress(prefix: str = "") -> ProgressCallback:
    def _cb(fraction: float, message: str) -> None:
        print(f"{prefix}[{fraction * 100:5.1f}%] {message}", flush=True)

    return _cb


def sub_progress(parent: Optional[ProgressCallback], start: float, end: float) -> ProgressCallback:
    """상위 진행률 [start, end] 구간에 매핑되는 하위 콜백."""
    if parent is None:
        return null_progress

    def _cb(fraction: float, message: str) -> None:
        parent(start + (end - start) * max(0.0, min(1.0, fraction)), message)

    return _cb
