"""결과 내보내기: CSV / Markdown / LaTeX(booktabs) — 논문 표 초안 생성."""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from core.experiments.stats import CORE_METRICS, HIGHER_IS_BETTER, summarize, summarize_pretty
from core.types import RunRecord

METRIC_LABELS = {
    "coherence_emb": "Coherence(emb)",
    "npmi": "NPMI",
    "diversity": "Diversity",
    "silhouette": "Silhouette",
    "separation": "Separation",
    "coverage": "Coverage",
    "nmi": "NMI(외적)",
    "ari": "ARI(외적)",
    "purity": "Purity(외적)",
    "n_topics_found": "#Topics",
    "runtime_sec": "Runtime(s)",
    "judge_clarity": "Judge-명확성",
    "judge_consistency": "Judge-일관성",
    "judge_distinctiveness": "Judge-구분성",
    "judge_groundedness": "Judge-근거성",
    "judge_overall": "Judge-종합",
}


def to_markdown_table(records: List[RunRecord], metrics: Optional[List[str]] = None) -> str:
    df = summarize_pretty(records, metrics)
    if df.empty:
        return "(결과 없음)"
    df = df.rename(columns=METRIC_LABELS)
    return df.to_markdown(index=False)


def to_csv(records: List[RunRecord], metrics: Optional[List[str]] = None) -> str:
    df = summarize(records, metrics)
    return df.to_csv(index=False)


def to_latex_table(
    records: List[RunRecord],
    metrics: Optional[List[str]] = None,
    caption: str = "토픽모델별 성능 비교 (평균 ± 표준편차)",
    label: str = "tab:topic-model-comparison",
) -> str:
    """booktabs 스타일 LaTeX 표. 최고값은 \\textbf 처리."""
    summary = summarize(records, metrics)
    if summary.empty:
        return "% (결과 없음)"
    metric_cols = [m for m in (metrics or CORE_METRICS) if f"{m}_mean" in summary.columns]

    header = ["Model"] + [METRIC_LABELS.get(m, m) for m in metric_cols]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(metric_cols) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        cells = [str(row["model"])]
        for m in metric_cols:
            mean = row.get(f"{m}_mean")
            std = row.get(f"{m}_std", 0.0) or 0.0
            if pd.isna(mean):
                cells.append("--")
                continue
            means = summary[f"{m}_mean"].dropna()
            best = means.max() if m in HIGHER_IS_BETTER else means.min()
            text = f"{mean:.3f} $\\pm$ {std:.3f}"
            if abs(mean - best) < 1e-12:
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def paired_tests_markdown(tests_df: pd.DataFrame, proposed: str, metric: str) -> str:
    if tests_df.empty:
        return "(검정 결과 없음)"
    lines = [f"**{proposed}** vs 베이스라인 — 지표: `{metric}` (시드 쌍체)", ""]
    lines.append(tests_df.round(4).to_markdown(index=False))
    return "\n".join(lines)
