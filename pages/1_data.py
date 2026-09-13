"""페이지 1: 데이터 준비 — BigKinds 헤드라인 / 논문 업로드 → 토큰화·임베딩·저장."""
import io

import pandas as pd
import streamlit as st

from core.data.bigkinds import load_bigkinds
from core.data.corpus_store import delete_corpus, list_corpora
from core.data.papers import DEFAULT_CHUNK_CHARS, DEFAULT_OVERLAP_CHARS, load_papers
from core.data.prepare import prepare_corpus
from pages._shared import render_ollama_status, st_progress_callback

st.title("📂 데이터 준비")
st.caption("두 데이터 체제: **헤드라인(다수 짧은 텍스트)** vs **논문(소수 긴 텍스트 → 청크)**. "
           "준비 시 Kiwi 토큰화와 bge-m3 임베딩을 사전 계산해 이후 실험을 빠르게 만듭니다.")
render_ollama_status()

tab_headlines, tab_mcpanal, tab_papers, tab_saved = st.tabs(
    ["📰 헤드라인 (BigKinds)", "🗂 MCPanal 월별 세션", "📄 논문 (PDF/DOCX)", "💾 저장된 코퍼스"]
)

with tab_headlines:
    uploaded = st.file_uploader("BigKinds Excel/CSV 업로드", type=["csv", "xlsx", "xls"], key="bk_upload")
    col1, col2, col3 = st.columns(3)
    with col1:
        corpus_name = st.text_input("코퍼스 이름", value="", placeholder="(비우면 파일명)", key="bk_name")
    with col2:
        min_length = st.number_input("최소 헤드라인 길이", 1, 50, 5, key="bk_minlen")
    with col3:
        dedup = st.checkbox("중복 헤드라인 제거", value=True, key="bk_dedup")

    use_sample = st.checkbox("샘플 데이터 사용 (data/samples/sample_bigkinds.csv)", key="bk_sample")

    if st.button("헤드라인 코퍼스 준비", type="primary", key="bk_prepare", disabled=not (uploaded or use_sample)):
        try:
            if use_sample:
                from config import DATA_DIR

                path = DATA_DIR / "samples" / "sample_bigkinds.csv"
                corpus = load_bigkinds(path, filename=path.name, min_length=min_length,
                                       dedup=dedup, name=corpus_name or "sample_headlines")
            else:
                corpus = load_bigkinds(io.BytesIO(uploaded.getvalue()), filename=uploaded.name,
                                       min_length=min_length, dedup=dedup, name=corpus_name)
            st.info(f"파싱 완료: {len(corpus)}건 (제목 열: {corpus.meta.get('title_column')})")
            placeholder = st.empty()
            prepare_corpus(corpus, progress=st_progress_callback(placeholder))
            placeholder.empty()
            st.success(f"저장 완료 — 코퍼스 해시 `{corpus.content_hash}` · {len(corpus)}건")
            st.dataframe(pd.DataFrame({"헤드라인": corpus.texts[:20]}), use_container_width=True)
        except Exception as e:
            st.error(f"준비 실패: {e}")

with tab_mcpanal:
    st.caption("MCPanal 경기도 언론분석에 저장된 월별 '대한민국 전체' 기사 세션을 가져옵니다. "
               "ChromaDB에 저장된 bge-m3 임베딩을 그대로 재사용하므로 **재임베딩이 없습니다** (토큰화만 수행).")
    from core.data import mcpanal_import as mp
    from core.data.corpus_store import save_embeddings

    if not mp.registry_available():
        st.warning(f"MCPanal 레지스트리를 찾을 수 없습니다: {mp.REGISTRY_PATH}")
    else:
        sessions = mp.list_monthly_sessions()
        if not sessions:
            st.info("임베딩 완료된 월별 세션이 없습니다.")
        else:
            options = {
                f"{s['period_label']} · {s['document_count']:,}건 · {s['session_id']}": s
                for s in sessions
            }
            chosen = st.multiselect(
                "가져올 월 선택 (복수 선택 시 하나의 코퍼스로 병합)",
                list(options.keys()),
                key="mp_sessions",
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                mp_name = st.text_input("코퍼스 이름", value="", placeholder="(비우면 기간 기반 자동)", key="mp_name")
            with col2:
                mp_minlen = st.number_input("최소 헤드라인 길이", 1, 50, 5, key="mp_minlen")
            with col3:
                mp_dedup = st.checkbox("중복 헤드라인 제거", value=True, key="mp_dedup")

            total_docs = sum(options[c]["document_count"] for c in chosen)
            if chosen:
                st.info(f"선택: {len(chosen)}개월, 약 {total_docs:,}건 "
                        "(토큰화에 수 분이 걸릴 수 있습니다)")
            if st.button("월별 세션 가져오기", type="primary", key="mp_import", disabled=not chosen):
                try:
                    placeholder = st.empty()
                    cb = st_progress_callback(placeholder)
                    corpus, embeddings = mp.load_monthly_corpus(
                        [options[c] for c in chosen],
                        name=mp_name, dedup=mp_dedup, min_length=int(mp_minlen),
                        progress=lambda f, m: cb(f * 0.3, m),
                    )
                    # 가져온 임베딩을 먼저 저장 → prepare_corpus가 재임베딩을 건너뜀
                    save_embeddings(corpus.content_hash, embeddings)
                    prepare_corpus(corpus, progress=lambda f, m: cb(0.3 + f * 0.7, m))
                    placeholder.empty()
                    st.success(f"저장 완료 — 코퍼스 해시 `{corpus.content_hash}` · {len(corpus):,}건 "
                               f"(기간: {', '.join(corpus.meta['periods'])})")
                    st.dataframe(
                        pd.DataFrame({
                            "날짜": [d.meta.get("date", "") for d in corpus.docs[:20]],
                            "언론사": [d.meta.get("press", "") for d in corpus.docs[:20]],
                            "헤드라인": [d.text for d in corpus.docs[:20]],
                        }),
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"가져오기 실패: {e}")
                    st.exception(e)

with tab_papers:
    files = st.file_uploader("논문 파일 업로드 (PDF/DOCX/TXT, 복수 가능)",
                             type=["pdf", "docx", "txt", "md"], accept_multiple_files=True, key="pp_upload")
    col1, col2, col3 = st.columns(3)
    with col1:
        paper_name = st.text_input("코퍼스 이름", value="papers", key="pp_name")
    with col2:
        chunk_chars = st.number_input("청크 크기(자)", 200, 4000, DEFAULT_CHUNK_CHARS, step=100, key="pp_chunk")
    with col3:
        overlap = st.number_input("청크 겹침(자)", 0, 500, DEFAULT_OVERLAP_CHARS, step=50, key="pp_overlap")
    clean = st.checkbox("논문 정제 (참고문헌/캡션/페이지번호 제거)", value=True, key="pp_clean")

    if st.button("논문 코퍼스 준비", type="primary", key="pp_prepare", disabled=not files):
        try:
            corpus = load_papers(
                [(f.getvalue(), f.name) for f in files],
                chunk_chars=int(chunk_chars), overlap_chars=int(overlap),
                clean=clean, name=paper_name,
            )
            per_file = corpus.meta.get("per_file_chunks", {})
            st.info("청킹 결과: " + ", ".join(f"{k}: {v}청크" for k, v in per_file.items()))
            placeholder = st.empty()
            prepare_corpus(corpus, progress=st_progress_callback(placeholder))
            placeholder.empty()
            st.success(f"저장 완료 — 코퍼스 해시 `{corpus.content_hash}` · {len(corpus)}청크")
            st.dataframe(
                pd.DataFrame({
                    "출처": [d.source_id for d in corpus.docs[:20]],
                    "청크 미리보기": [d.text[:150] for d in corpus.docs[:20]],
                }),
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"준비 실패: {e}")

with tab_saved:
    corpora = list_corpora()
    if not corpora:
        st.info("저장된 코퍼스가 없습니다.")
    for meta in corpora:
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f"**{meta['name']}** · {meta['regime']} · {meta['n_docs']}건 · "
                    f"`{meta['content_hash']}` · 저장: {meta.get('saved_at', '?')}"
                )
            with col2:
                if st.button("삭제", key=f"del_{meta['content_hash']}"):
                    delete_corpus(meta["content_hash"])
                    st.rerun()
