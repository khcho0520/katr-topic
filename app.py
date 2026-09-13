"""TopicNew — 토픽모델링 개선 모델 검증 플랫폼 (Streamlit 진입점)."""
import streamlit as st

st.set_page_config(
    page_title="TopicNew — 토픽모델링 검증",
    page_icon="🧪",
    layout="wide",
)

pages = st.navigation(
    [
        st.Page("pages/1_data.py", title="1. 데이터 준비", icon="📂"),
        st.Page("pages/2_explore.py", title="2. 단일 실행 탐색", icon="🔍"),
        st.Page("pages/3_experiment.py", title="3. 배치 실험", icon="🧪"),
        st.Page("pages/4_results.py", title="4. 결과·통계·내보내기", icon="📊"),
    ]
)
pages.run()
