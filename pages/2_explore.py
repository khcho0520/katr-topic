"""페이지 2: 단일 실행 탐색 — 모델 1개 × 시드 1개 실행, 세션 저장/복원 지원."""
import dataclasses

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.data.corpus_store import corpus_exists
from core.experiments import workspace as ws
from core.metrics.suite import evaluate
from core.models import PRESETS, create_model
from core.models.qualit.pipeline import QualITConfig
from pages._shared import (
    cached_embeddings,
    cached_load_corpus,
    render_ollama_status,
    select_corpus,
    st_progress_callback,
)

st.title("🔍 단일 실행 탐색")
st.caption("변형 아이디어를 빠르게 실험하는 페이지입니다. 논문용 반복실험은 **3. 배치 실험**에서. "
           "결과는 아래에서 세션으로 저장해 재시작 후에도 복원할 수 있습니다.")
render_ollama_status()

# ---- 저장된 세션 (복원/삭제) ----
saved_sessions = ws.list_workspaces()
with st.expander(f"💾 저장된 세션 ({len(saved_sessions)}개)", expanded=False):
    if not saved_sessions:
        st.info("저장된 세션이 없습니다. 실행 후 결과 하단에서 저장하세요.")
    else:
        options = {
            f"{w['name']} · {w['model']} · seed={w['seed']} · {w['corpus_name']} · {w['created_at']}": w["workspace_id"]
            for w in saved_sessions
        }
        chosen = st.selectbox("세션 선택", list(options.keys()), key="ws_select")
        col_load, col_delete = st.columns([1, 1])
        with col_load:
            if st.button("📂 불러오기", key="ws_load"):
                loaded = ws.load_workspace(options[chosen])
                if loaded is None:
                    st.error("세션 파일을 찾을 수 없습니다.")
                else:
                    st.session_state["explore_view"] = {
                        "result": loaded["result"],
                        "report": loaded["report"],
                        "corpus_hash": loaded["corpus_hash"],
                        "config": loaded["config"],
                        "source": f"세션 '{loaded['name']}' ({loaded['created_at']})",
                    }
                    st.rerun()
        with col_delete:
            if st.button("🗑️ 삭제", key="ws_delete"):
                ws.delete_workspace(options[chosen])
                st.rerun()

# ---- 코퍼스·모델 설정 ----
meta = select_corpus("explore_corpus")
if meta is None and "explore_view" not in st.session_state:
    st.stop()

col1, col2 = st.columns([2, 1])
with col1:
    preset_key = st.selectbox(
        "모델 프리셋",
        list(PRESETS.keys()),
        format_func=lambda k: f"{PRESETS[k].label}  ({k})",
    )
with col2:
    seed = st.number_input("시드", 0, 9999, 42)

config = PRESETS[preset_key]

# QualIT 계열이면 ablation 필드 편집 허용
if isinstance(config, QualITConfig):
    with st.expander("⚙️ QualIT 파이프라인 설정 (ablation)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            halluc = st.selectbox("환각 필터", ["cosine", "percentile", "llm_verify", "none"],
                                  index=["cosine", "percentile", "llm_verify", "none"].index(config.hallucination_filter))
            halluc_th = st.number_input("환각 임계값", 0.0, 1.0, config.hallucination_threshold, 0.01)
            kp_model = st.text_input("키프레이즈 LLM", config.keyphrase_model)
        with c2:
            rep = st.selectbox("클러스터링 표현", ["keyphrase", "doc", "joint"],
                               index=["keyphrase", "doc", "joint"].index(config.representation))
            alpha = st.slider("joint α (문서 비중)", 0.0, 1.0, config.joint_alpha, 0.05)
            clustering = st.selectbox("클러스터링", ["two_level_kmeans", "kmeans", "hdbscan"],
                                      index=["two_level_kmeans", "kmeans", "hdbscan"].index(config.clustering))
        with c3:
            keywords = st.selectbox("키워드 방식", ["llm", "ctfidf"],
                                    index=["llm", "ctfidf"].index(config.keywords))
            refine = st.checkbox("유사 토픽 병합", config.llm_refine)
            max_topics = st.number_input("최대 토픽 수", 2, 30, config.max_topics)
            max_llm_docs = st.number_input(
                "LLM 대상 대표 문서 상한 (0=무제한)", 0, 20000, config.max_llm_docs, step=100,
                help="대량 코퍼스 보호: 초과분은 임베딩 층화 샘플링으로 대표만 추출하고 "
                     "나머지 문서는 최근접 토픽에 할당됩니다.",
            )
        config = dataclasses.replace(
            config,
            hallucination_filter=halluc, hallucination_threshold=halluc_th,
            keyphrase_model=kp_model, representation=rep, joint_alpha=alpha,
            clustering=clustering, keywords=keywords, llm_refine=refine,
            max_topics=int(max_topics), max_llm_docs=int(max_llm_docs),
        )
else:
    n_topics = st.number_input(
        "토픽 수 (BERTopic은 0=자동)", 0, 50,
        int(config.n_topics or 0),
    )
    if n_topics > 0:
        config = dataclasses.replace(config, n_topics=int(n_topics))

if meta is not None and st.button("▶️ 실행", type="primary"):
    corpus = cached_load_corpus(meta["content_hash"])
    embeddings = cached_embeddings(meta["content_hash"])
    placeholder = st.empty()
    try:
        model = create_model(config)
        result = model.fit(corpus, embeddings, int(seed), progress=st_progress_callback(placeholder))
        placeholder.empty()
        report = evaluate(result, corpus, embeddings)
        st.session_state["explore_view"] = {
            "result": result,
            "report": report,
            "corpus_hash": meta["content_hash"],
            "config": config.to_dict(),
            "source": "방금 실행",
        }
    except Exception as e:
        placeholder.empty()
        st.error(f"실행 실패: {e}")
        st.exception(e)

# ---- 결과 렌더링 (실행 결과 또는 복원된 세션) ----
view = st.session_state.get("explore_view")
if view is None:
    st.stop()

result = view["result"]
report = view["report"]
view_corpus_hash = view["corpus_hash"]

st.divider()
st.subheader(f"결과 — {result.model_name} (seed={result.seed}, {result.runtime_sec:.1f}s)")
st.caption(f"출처: {view['source']} · 코퍼스 `{view_corpus_hash[:12]}`")
with st.expander("이 결과의 설정 스냅샷", expanded=False):
    st.json(view["config"])

# 지표 카드
scalars = report.scalars
cols = st.columns(6)
for col, (key, label) in zip(cols, [
    ("coherence_emb", "Coherence"), ("npmi", "NPMI"), ("diversity", "Diversity"),
    ("silhouette", "Silhouette"), ("coverage", "Coverage"), ("n_topics_found", "#Topics"),
]):
    value = scalars.get(key)
    col.metric(label, f"{value:.3f}" if value is not None else "—")

# 토픽 표
if report.per_topic is not None and not report.per_topic.empty:
    st.dataframe(report.per_topic, use_container_width=True, hide_index=True)
    fig = px.bar(report.per_topic, x="topic_id", y="size", hover_data=["label"],
                 title="토픽별 문서 수")
    st.plotly_chart(fig, use_container_width=True)

# ---- 세션 저장 ----
with st.container(border=True):
    st.markdown("**💾 이 결과를 세션으로 저장**")
    col_name, col_button = st.columns([3, 1])
    with col_name:
        session_name = st.text_input(
            "세션 이름", key="ws_save_name",
            placeholder="예: qualit_hdbscan_헤드라인_1차",
            label_visibility="collapsed",
        )
    with col_button:
        if st.button("저장", type="primary", key="ws_save"):
            try:
                corpus_for_save = cached_load_corpus(view_corpus_hash) if corpus_exists(view_corpus_hash) else None
                workspace_id = ws.save_workspace(
                    name=session_name or f"{result.model_name}_s{result.seed}",
                    corpus_hash=view_corpus_hash,
                    corpus_name=corpus_for_save.name if corpus_for_save else "",
                    regime=corpus_for_save.regime if corpus_for_save else "",
                    config_dict=view["config"],
                    seed=result.seed,
                    result=result,
                    report=report,
                )
                st.success(f"세션 저장됨: `{workspace_id}` (재시작 후에도 위 '저장된 세션'에서 복원 가능)")
            except Exception as e:
                st.error(f"저장 실패: {e}")

# ---- 코퍼스 의존 뷰 (원본 코퍼스가 남아 있을 때만) ----
if not corpus_exists(view_corpus_hash):
    st.warning("이 세션의 원본 코퍼스가 삭제되어 문서 보기·UMAP 지도는 표시할 수 없습니다.")
    st.stop()

view_corpus = cached_load_corpus(view_corpus_hash)
view_embeddings = cached_embeddings(view_corpus_hash)

with st.expander("🗺️ 2D 임베딩 지도 (UMAP)", expanded=False):
    try:
        from umap import UMAP

        @st.cache_data(show_spinner="UMAP 투영 중...")
        def _project(corpus_hash: str) -> np.ndarray:
            embeddings = cached_embeddings(corpus_hash)
            return UMAP(n_components=2, random_state=42, metric="cosine").fit_transform(embeddings)

        proj = _project(view_corpus_hash)
        df = pd.DataFrame({
            "x": proj[:, 0], "y": proj[:, 1],
            "topic": [str(int(a)) for a in result.assignments],
            "label": [result.topic_labels.get(int(a), "아웃라이어") for a in result.assignments],
            "text": [t[:80] for t in view_corpus.texts],
        })
        fig2 = px.scatter(df, x="x", y="y", color="topic", hover_data=["label", "text"], height=550)
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.warning(f"UMAP 투영 실패: {e}")

with st.expander("📄 토픽별 문서 보기", expanded=False):
    if result.topic_ids:
        tid = st.selectbox("토픽 선택", result.topic_ids,
                           format_func=lambda t: f"{t}: {result.topic_labels.get(t, '')}")
        member_idx = np.where(result.assignments == tid)[0]
        st.dataframe(
            pd.DataFrame({
                "doc_id": [view_corpus.docs[i].doc_id for i in member_idx],
                "텍스트": [view_corpus.docs[i].text[:200] for i in member_idx],
            }),
            use_container_width=True, hide_index=True,
        )
