"""BigKinds(뉴스 빅데이터) Excel/CSV → Corpus. 헤드라인 1건 = 문서 1건."""
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from core.types import Corpus, Document
from core.utils.hashing import texts_hash

# BigKinds 및 일반 뉴스 데이터의 제목 열 후보 (우선순위 순)
TITLE_COLUMN_CANDIDATES = ["제목", "기사제목", "뉴스제목", "헤드라인", "title", "headline"]
DATE_COLUMN_CANDIDATES = ["일자", "날짜", "date", "발행일"]
PRESS_COLUMN_CANDIDATES = ["언론사", "매체", "press", "publisher"]
CATEGORY_COLUMN_CANDIDATES = ["통합 분류1", "통합분류1", "분류", "카테고리", "category"]


def _read_table(source: Union[str, Path, io.BytesIO], filename: str = "") -> pd.DataFrame:
    name = (filename or str(source)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(source)
    # CSV: BigKinds는 UTF-8 또는 CP949 혼재
    try:
        return pd.read_csv(source)
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, encoding="cp949")


def detect_title_column(df: pd.DataFrame) -> Optional[str]:
    columns = {str(c).strip(): c for c in df.columns}
    for cand in TITLE_COLUMN_CANDIDATES:
        for norm, original in columns.items():
            if norm.lower() == cand.lower():
                return original
    # 폴백: 문자열 비율이 가장 높고 평균 길이 10자 이상인 열
    best, best_len = None, 0.0
    for col in df.columns:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        avg_len = series.str.len().mean()
        if avg_len >= 10 and avg_len > best_len:
            best, best_len = col, avg_len
    return best


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    columns = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in columns:
            return columns[cand.lower()]
    return None


def load_bigkinds(
    source: Union[str, Path, io.BytesIO],
    filename: str = "",
    title_column: Optional[str] = None,
    min_length: int = 5,
    dedup: bool = True,
    name: str = "",
) -> Corpus:
    df = _read_table(source, filename)
    title_col = title_column or detect_title_column(df)
    if title_col is None:
        raise ValueError(f"제목 열을 찾지 못했습니다. 열 목록: {list(df.columns)}")

    date_col = _find_column(df, DATE_COLUMN_CANDIDATES)
    press_col = _find_column(df, PRESS_COLUMN_CANDIDATES)
    category_col = _find_column(df, CATEGORY_COLUMN_CANDIDATES)

    docs: List[Document] = []
    seen: set[str] = set()
    for row_idx, row in df.iterrows():
        text = str(row[title_col]).strip() if pd.notna(row[title_col]) else ""
        if len(text) < min_length:
            continue
        if dedup:
            if text in seen:
                continue
            seen.add(text)
        meta = {}
        if date_col and pd.notna(row.get(date_col)):
            meta["date"] = str(row[date_col])
        if press_col and pd.notna(row.get(press_col)):
            meta["press"] = str(row[press_col])
        if category_col and pd.notna(row.get(category_col)):
            meta["category"] = str(row[category_col])   # 외적 타당도(NMI/ARI) 정답 레이블
        docs.append(
            Document(
                doc_id=f"h{len(docs)}",
                text=text,
                source_id=f"{filename or 'bigkinds'}:{row_idx}",
                meta=meta,
            )
        )

    if not docs:
        raise ValueError("유효한 헤드라인이 없습니다 (길이/중복 필터 확인).")

    corpus_name = name or (Path(filename).stem if filename else "headlines")
    return Corpus(
        regime="headlines",
        docs=docs,
        content_hash=texts_hash([d.text for d in docs]),
        name=corpus_name,
        meta={
            "title_column": str(title_col),
            "n_rows": int(len(df)),
            "n_docs": len(docs),
            "dedup": dedup,
            "min_length": min_length,
        },
    )
