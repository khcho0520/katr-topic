"""페이지 4: 결과 브라우저 — 평균±표준편차 표, 분포 플롯, 유의성 검정, 내보내기."""
import pandas as pd
import plotly.express as px
import streamlit as st

from core.experiments import store
from core.experiments.export import (
    METRIC_LABELS,
    paired_tests_markdown,
    to_csv,
    to_latex_table,
    to_markdown_table,
)
from core.experiments.stats import CORE_METRICS, paired_tests, stability_table, summarize_pretty
from pages._shared import render_ollama_status

st.title("📊 결과 · 통계 · 내보내기")
render_ollama_status()

experiments = store.list_experiments()
if not experiments:
    st.warning("실행된 실험이 없습니다. **3. 배치 실험**에서 먼저 실험을 실행하세요.")
    st.stop()

exp_options = {
    f"{e['experiment_id']} · {e['n_runs']} runs · {e['corpus_name']}": e["experiment_id"]
    for e in experiments
}
selected = st.selectbox("실험 선택", list(exp_options.keys()))
experiment_id = exp_options[selected]

records = store.load_experiment_runs(experiment_id)
done_records = [r for r in records if r.status == "done"]
failed = [r for r in records if r.status == "failed"]
if failed:
    with st.expander(f"⚠️ 실패한 run {len(failed)}건", expanded=False):
        for r in failed:
            st.text(f"{r.run_id}: {r.error[:200]}")
if not done_records:
    st.error("완료된 run이 없습니다.")
    st.stop()

df = store.runs_to_dataframe(done_records)
models = sorted(df["model"].unique().tolist())
available_metrics = [m for m in CORE_METRICS if m in df.columns] + \
    [c for c in df.columns if c.startswith("judge_")]

# ---- 요약 표 ----
st.subheader("모델별 성능 요약 (평균 ± 표준편차, 굵게 = 최고값)")
pretty = summarize_pretty(done_records)
st.dataframe(pretty.rename(columns=METRIC_LABELS), use_container_width=True, hide_index=True)

# ---- 시드 간 안정성 ----
stability = stability_table(done_records)
if not stability.empty:
    st.subheader("시드 간 안정성 (재현 안정성)")
    st.caption("같은 설정의 시드 쌍별 할당 ARI — 1.0이면 시드가 바뀌어도 동일한 토픽 구조. "
               "낮으면 초기화에 민감한 방법이라는 뜻입니다. (두 시드 모두 할당된 문서만 비교)")
    st.dataframe(stability.round(4), use_container_width=True, hide_index=True)
    st.download_button("⬇️ 안정성 CSV", stability.to_csv(index=False),
                       file_name=f"{experiment_id}_stability.csv", mime="text/csv")
else:
    st.caption("ℹ️ 시드 간 안정성: 할당 원자료가 저장된 run이 설정당 2개 이상일 때 표시됩니다 "
               "(이 기능 추가 이후 실행된 실험부터 저장됨).")

# ---- 분포 플롯 ----
st.subheader("지표 분포 (시드별)")
metric = st.selectbox("지표", available_metrics, format_func=lambda m: METRIC_LABELS.get(m, m))
fig = px.box(df, x="model", y=metric, points="all", color="model",
             title=f"{METRIC_LABELS.get(metric, metric)} — 시드별 분포")
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ---- 유의성 검정 ----
st.subheader("쌍체 유의성 검정 (제안 모델 vs 베이스라인)")
col1, col2 = st.columns(2)
with col1:
    proposed = st.selectbox("제안 모델", models,
                            index=next((i for i, m in enumerate(models) if "qualit" in m.lower() or "QualIT" in m), 0))
with col2:
    baselines = st.multiselect("비교 대상", [m for m in models if m != proposed],
                               default=[m for m in models if m != proposed])
if baselines:
    tests = paired_tests(done_records, proposed, baselines, metric)
    if not tests.empty:
        st.dataframe(tests.round(4), use_container_width=True, hide_index=True)
        st.caption("t_p/wilcoxon_p < 0.05 → 유의. holm_p = 다중비교(Holm) 보정. "
                   "cohen_d: 0.2 작음 / 0.5 중간 / 0.8 큼. 쌍체 = 동일 시드.")
    else:
        st.info("공통 시드가 2개 이상인 쌍이 없습니다.")

# ---- 토픽 상세 ----
with st.expander("🔎 run별 토픽 상세", expanded=False):
    run_ids = [r.run_id for r in done_records]
    rid = st.selectbox("run 선택", run_ids)
    rec = next(r for r in done_records if r.run_id == rid)
    rows = [
        {"topic_id": tid, "label": t["label"], "size": t["size"], "keywords": ", ".join(t["keywords"])}
        for tid, t in sorted(rec.topics.items(), key=lambda kv: -kv[1]["size"])
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---- 내보내기 ----
st.subheader("내보내기 (논문 표 초안)")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.download_button("⬇️ 요약 CSV", to_csv(done_records),
                       file_name=f"{experiment_id}_summary.csv", mime="text/csv")
with c2:
    st.download_button("⬇️ Markdown 표", to_markdown_table(done_records),
                       file_name=f"{experiment_id}_table.md", mime="text/markdown")
with c3:
    st.download_button("⬇️ LaTeX 표", to_latex_table(done_records),
                       file_name=f"{experiment_id}_table.tex", mime="text/plain")
with c4:
    raw_csv = df.to_csv(index=False)
    st.download_button("⬇️ 전체 run 원자료 CSV", raw_csv,
                       file_name=f"{experiment_id}_runs.csv", mime="text/csv")

if baselines:
    tests = paired_tests(done_records, proposed, baselines, metric)
    if not tests.empty:
        st.download_button(
            "⬇️ 유의성 검정 Markdown",
            paired_tests_markdown(tests, proposed, metric),
            file_name=f"{experiment_id}_tests_{metric}.md",
            mime="text/markdown",
        )
