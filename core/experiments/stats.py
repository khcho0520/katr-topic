"""논문용 통계: 모델×지표 요약(평균±표준편차), 쌍체 검정, 효과크기, 다중비교 보정.

쌍체 구성: 동일 코퍼스·동일 시드의 두 모델 결과가 한 쌍 (paired samples).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

from core.types import RunRecord

# 높을수록 좋은 지표 (표에서 최고값 강조용)
HIGHER_IS_BETTER = {
    "coherence_emb", "separation", "silhouette", "silhouette_noise",
    "npmi", "diversity", "coverage",
    "nmi", "ari", "purity", "stability_ari",
    "judge_clarity", "judge_consistency", "judge_distinctiveness",
    "judge_groundedness", "judge_overall",
}
CORE_METRICS = [
    "coherence_emb", "npmi", "diversity", "silhouette", "separation",
    "nmi", "ari", "purity",
    "coverage", "n_topics_found", "runtime_sec",
]


def _records_frame(records: List[RunRecord]) -> pd.DataFrame:
    from core.experiments.store import runs_to_dataframe

    df = runs_to_dataframe([r for r in records if r.status == "done"])
    return df


def summarize(records: List[RunRecord], metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """모델별 지표 평균±표준편차 표. 행=모델, 열=지표."""
    df = _records_frame(records)
    if df.empty:
        return pd.DataFrame()
    metric_cols = metrics or [m for m in CORE_METRICS if m in df.columns]
    metric_cols += [c for c in df.columns if c.startswith("judge_") and c not in metric_cols]
    rows = []
    for model, group in df.groupby("model", sort=False):
        row: Dict[str, object] = {"model": model, "n_runs": len(group)}
        for m in metric_cols:
            values = group[m].dropna()
            if values.empty:
                continue
            row[f"{m}_mean"] = float(values.mean())
            row[f"{m}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_pretty(records: List[RunRecord], metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """'평균±표준편차' 문자열 표 + 최고값 굵게(마크다운) — 논문 표 초안용."""
    df = _records_frame(records)
    if df.empty:
        return pd.DataFrame()
    metric_cols = metrics or [m for m in CORE_METRICS if m in df.columns]
    metric_cols += [c for c in df.columns if c.startswith("judge_") and c not in metric_cols]

    summary = summarize(records, metric_cols)
    out_rows = []
    for _, row in summary.iterrows():
        pretty: Dict[str, object] = {"model": row["model"], "n_runs": row["n_runs"]}
        for m in metric_cols:
            mean_key = f"{m}_mean"
            if mean_key not in summary.columns or pd.isna(row.get(mean_key)):
                continue
            mean, std = row[mean_key], row.get(f"{m}_std", 0.0) or 0.0
            means = summary[mean_key].dropna()
            best = means.max() if m in HIGHER_IS_BETTER else means.min()
            text = f"{mean:.4f} ± {std:.4f}"
            if np.isclose(mean, best):
                text = f"**{text}**"
            pretty[m] = text
        out_rows.append(pretty)
    return pd.DataFrame(out_rows)


def stability_table(records: List[RunRecord]) -> pd.DataFrame:
    """시드 간 안정성: 같은 설정의 시드 쌍별 할당 ARI (재현 안정성 지표, 높을수록 안정).

    두 시드 모두에서 할당된(-1 아님) 문서만 비교한다. 할당 원자료(assignments)가
    저장된 run만 대상 (과거 기록에는 없을 수 있음).
    """
    from itertools import combinations

    from sklearn.metrics import adjusted_rand_score

    rows = []
    done = [r for r in records if r.status == "done" and r.assignments]
    by_config: Dict[tuple, List[RunRecord]] = {}
    for r in done:
        by_config.setdefault((r.config.get("display_name") or r.model_name, r.config_hash), []).append(r)

    for (model, _config_hash), runs in by_config.items():
        if len(runs) < 2:
            continue
        aris = []
        for r1, r2 in combinations(sorted(runs, key=lambda r: r.seed), 2):
            a = np.array(r1.assignments)
            b = np.array(r2.assignments)
            if len(a) != len(b):
                continue
            mask = (a >= 0) & (b >= 0)
            if mask.sum() < 10:
                continue
            aris.append(adjusted_rand_score(a[mask], b[mask]))
        if not aris:
            continue
        rows.append(
            {
                "model": model,
                "n_seeds": len(runs),
                "n_pairs": len(aris),
                "stability_ari_mean": float(np.mean(aris)),
                "stability_ari_std": float(np.std(aris, ddof=1)) if len(aris) > 1 else 0.0,
                "stability_ari_min": float(np.min(aris)),
            }
        )
    return pd.DataFrame(rows).sort_values("stability_ari_mean", ascending=False) if rows else pd.DataFrame()


def paired_tests(
    records: List[RunRecord],
    proposed: str,
    baselines: List[str],
    metric: str,
) -> pd.DataFrame:
    """제안 모델 vs 각 베이스라인의 시드-쌍체 검정.

    반환 열: baseline, n_pairs, mean_diff, t_stat, t_p, wilcoxon_p, cohen_d, holm_p(t 기준)
    """
    df = _records_frame(records)
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    prop = df[df["model"] == proposed].set_index("seed")[metric].dropna()
    rows = []
    for baseline in baselines:
        base = df[df["model"] == baseline].set_index("seed")[metric].dropna()
        common_seeds = sorted(set(prop.index) & set(base.index))
        if len(common_seeds) < 2:
            rows.append({"baseline": baseline, "n_pairs": len(common_seeds)})
            continue
        a = prop.loc[common_seeds].to_numpy(dtype=float)
        b = base.loc[common_seeds].to_numpy(dtype=float)
        diff = a - b
        t_stat, t_p = sps.ttest_rel(a, b)
        try:
            if np.allclose(diff, 0):
                w_p = 1.0
            else:
                _, w_p = sps.wilcoxon(a, b)
        except Exception:
            w_p = float("nan")
        sd = diff.std(ddof=1)
        cohen_d = float(diff.mean() / sd) if sd > 0 else 0.0
        rows.append(
            {
                "baseline": baseline,
                "n_pairs": len(common_seeds),
                "mean_diff": float(diff.mean()),
                "t_stat": float(t_stat),
                "t_p": float(t_p),
                "wilcoxon_p": float(w_p),
                "cohen_d": cohen_d,
            }
        )
    result = pd.DataFrame(rows)
    # Holm 보정 (t_p 기준)
    if "t_p" in result.columns:
        valid = result["t_p"].notna()
        pvals = result.loc[valid, "t_p"].to_numpy()
        order = np.argsort(pvals)
        m = len(pvals)
        adjusted = np.empty(m)
        prev = 0.0
        for rank, idx in enumerate(order):
            adj = min(1.0, (m - rank) * pvals[idx])
            prev = max(prev, adj)
            adjusted[idx] = prev
        result.loc[valid, "holm_p"] = adjusted
    return result
