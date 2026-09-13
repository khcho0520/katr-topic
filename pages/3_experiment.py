"""페이지 3: 배치 실험 — N시드 × M모델설정 그리드 실행 (재개 가능)."""
import re

import streamlit as st

from core.experiments import store
from core.experiments.runner import make_run_id, run_experiment
from core.llm.ollama_client import get_default_client
from core.models import PRESETS, apply_common_conditions
from pages._shared import load_selected, render_ollama_status, select_corpus

st.title("🧪 배치 실험")
st.caption("논문용 반복실험: 동일 코퍼스에 (설정 × 시드) 그리드를 실행합니다. "
           "LLM 단계는 캐시로 시드 간 재사용되고, 중단해도 같은 실험 ID로 재개됩니다.")
render_ollama_status()

meta = select_corpus("exp_corpus")
if meta is None:
    st.stop()

default_exp_id = f"exp_{meta['name']}_{meta['content_hash'][:8]}"
experiment_id = st.text_input("실험 ID (같은 ID로 재실행하면 이어서 실행)", value=default_exp_id)
experiment_id = re.sub(r"[^\w\-.]", "_", experiment_id)

col1, col2 = st.columns([3, 2])
with col1:
    selected_presets = st.multiselect(
        "모델 설정 (프리셋)",
        list(PRESETS.keys()),
        default=["katr_core", "bertopic_full", "bertopic", "lda", "nmf", "qualit_paper"],
        format_func=lambda k: f"{PRESETS[k].label}  ({k})",
    )
with col2:
    seeds_text = st.text_input("시드 목록 (쉼표 구분)", value="42, 43, 44, 45, 46")
    judge = st.checkbox("LLM-as-judge 평가 포함 (느림, 캐시됨)", value=False)

try:
    seeds = [int(s.strip()) for s in seeds_text.split(",") if s.strip()]
except ValueError:
    st.error("시드는 정수 쉼표 구분으로 입력하세요.")
    st.stop()

configs = [PRESETS[k] for k in selected_presets]

# ---- 공통 조건 (조건 통일) ----
def _apply_recommendation() -> None:
    """자동 제안 계산 → 공통 조건 위젯 값 채우기 (on_click 콜백: 위젯 생성 전에 실행됨)."""
    from core.experiments.recommend import recommend_common_conditions

    corpus, embeddings = load_selected(meta)
    reco = recommend_common_conditions(corpus, embeddings)
    st.session_state["cc_k"] = reco.n_topics
    st.session_state["cc_mcs"] = reco.bertopic_min_cluster_size
    st.session_state["cc_llm"] = reco.qualit_max_llm_docs
    st.session_state["cc_reco"] = reco


with st.expander("⚖️ 공통 조건 — 모델 간 조건 통일 (K 교란 제거)", expanded=False):
    st.caption("논문용 공정 비교를 위해 선택한 모든 설정에 동일 조건을 일괄 적용합니다. "
               "적용된 설정은 결과 표에서 접미사(K=…, mcs=…)로 구분됩니다.")
    st.button(
        "🔮 데이터 기반 자동 제안",
        on_click=_apply_recommendation,
        help="코퍼스 임베딩 표본의 실루엣 k 스캔으로 공통 K를 추정하고, "
             "코퍼스 크기에 맞는 BERTopic mcs·QualIT LLM 상한을 함께 제안합니다 (결정적).",
    )
    reco = st.session_state.get("cc_reco")
    if reco is not None:
        st.info("**자동 제안 근거**\n" + "\n".join(f"- {line}" for line in reco.rationale))
        if reco.k_scores:
            st.caption("k별 실루엣: " + " · ".join(
                f"k={k}: {s:.3f}" for k, s in sorted(reco.k_scores.items())))

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        common_k = st.number_input(
            "공통 토픽 수 K (0=모델별 자동)", 0, 50, 0, key="cc_k",
            help="모든 모델에 동일 K 고정. LDA/NMF=성분 수, BERTopic=nr_topics 축소, "
                 "QualIT=클러스터링 목표 K. QualIT의 hdbscan 변형은 K-free라 미적용.",
        )
    with cc2:
        common_mcs = st.number_input(
            "BERTopic min_cluster_size (0=자동 n/20)", 0, 10000, 0, key="cc_mcs",
            help="기본 n/20은 대량 코퍼스에서 과대해져 토픽 수가 붕괴합니다. "
                 "K 고정 시 HDBSCAN이 K개 이상 찾도록 충분히 낮추세요 (예: 헤드라인 수만 건 → 50~200).",
        )
    with cc3:
        common_llm_docs = st.number_input(
            "QualIT LLM 대표 문서 상한 (-1=프리셋 기본)", -1, 20000, -1, step=100, key="cc_llm",
        )
    if common_k > 0 and common_mcs == 0 and any(c.name == "bertopic" for c in configs):
        st.warning("K를 고정했지만 BERTopic min_cluster_size가 자동(n/20)입니다 — "
                   "HDBSCAN이 K개보다 적게 찾으면 K 고정이 무의미해집니다. mcs를 함께 낮추는 것을 권장합니다.")

configs = apply_common_conditions(
    configs,
    n_topics=int(common_k),
    bertopic_min_cluster_size=int(common_mcs),
    qualit_max_llm_docs=None if common_llm_docs < 0 else int(common_llm_docs),
)
if any(c.label != PRESETS[k].label for c, k in zip(configs, selected_presets)):
    st.caption("적용된 설정: " + " · ".join(f"`{c.label}`" for c in configs))

# 실행 전 미리보기: 완료/대기 run 수
if configs and seeds:
    total = len(configs) * len(seeds)
    done = sum(
        1 for c in configs for s in seeds
        if store.run_exists(experiment_id, make_run_id(c, s))
    )
    st.info(f"그리드: {len(configs)}개 설정 × {len(seeds)}개 시드 = {total} runs "
            f"(이미 완료 {done}, 실행 대기 {total - done})")

if st.button("🚀 실험 실행", type="primary", disabled=not (configs and seeds)):
    corpus, embeddings = load_selected(meta)
    llm = get_default_client()
    llm_calls_before = llm.call_count

    progress_bar = st.progress(0.0, text="시작")
    log_area = st.container(height=300)
    records = []
    with st.status(f"실험 `{experiment_id}` 실행 중...", expanded=True) as status:
        def _cb(fraction: float, message: str) -> None:
            progress_bar.progress(fraction, text=message)

        for record in run_experiment(
            corpus, embeddings, configs, seeds, experiment_id,
            judge=judge, resume=True, progress=_cb,
        ):
            records.append(record)
            icon = "✅" if record.status == "done" else "❌"
            with log_area:
                model_label = record.config.get("display_name") or record.model_name
                st.text(f"{icon} {model_label} seed={record.seed} "
                        + (f"coherence={record.metrics.get('coherence_emb', 0):.3f} "
                           f"npmi={record.metrics.get('npmi', 0):.3f}" if record.status == "done"
                           else f"오류: {record.error[:100]}"))
        n_failed = sum(1 for r in records if r.status == "failed")
        status.update(
            label=f"완료: {len(records)} runs (실패 {n_failed}) · "
                  f"신규 LLM 호출 {llm.call_count - llm_calls_before}건",
            state="complete" if n_failed == 0 else "error",
        )
    st.success(f"실험 `{experiment_id}` 종료. **4. 결과·통계·내보내기** 페이지에서 확인하세요.")

# 기존 실험 현황
st.divider()
st.subheader("저장된 실험")
experiments = store.list_experiments()
if not experiments:
    st.info("아직 실행된 실험이 없습니다.")
else:
    for exp in experiments:
        st.markdown(
            f"- **{exp['experiment_id']}** · {exp['n_runs']} runs · "
            f"{exp['corpus_name']} ({exp['regime']}) · {exp['created_at']}"
        )
