import os
import subprocess
import sys

import streamlit as st

from src.dashboard import DashboardAnalytics
from src.summary import ActivitySummary

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="ChronoVisionAI",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
<style>

.block-container{
    padding-top:2rem;
    padding-left:2rem;
    padding-right:2rem;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

div[data-testid="stMetric"]{
    background:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 2px 8px rgba(0,0,0,.1);
}

</style>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# HEADER
# ==========================================================

st.title("🎥 ChronoVisionAI")
st.caption("AI Powered Human Activity Timeline Generator")
st.divider()

# ==========================================================
# SESSION STATE
# ==========================================================

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "processed_video" not in st.session_state:
    st.session_state.processed_video = None

if "last_upload_name" not in st.session_state:
    st.session_state.last_upload_name = None

# ==========================================================
# VIDEO UPLOAD
# ==========================================================

st.header("📤 Upload CCTV Video")

uploaded_file = st.file_uploader(
    "Choose a video",
    type=["mp4", "avi", "mov"],
)

video_path = None

if uploaded_file is not None:
    if st.session_state.last_upload_name != uploaded_file.name:
        st.session_state.analysis_complete = False
        st.session_state.processed_video = None
        st.session_state.last_upload_name = uploaded_file.name

    os.makedirs(os.path.join(PROJECT_ROOT, "videos"), exist_ok=True)

    video_path = os.path.join(PROJECT_ROOT, "videos", uploaded_file.name)

    with open(video_path, "wb") as video_file:
        video_file.write(uploaded_file.getbuffer())

    st.success("✅ Video Uploaded")
    st.video(video_path)

_, center, _ = st.columns([1, 2, 1])

with center:
    if st.button("🔍 Analyze Video", use_container_width=True):
        with st.spinner("Analyzing Video..."):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.activity",
                    video_path,
                ],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                env={
                    **os.environ,
                    "KMP_DUPLICATE_LIB_OK": "TRUE",
                },
            )

            if result.returncode == 0:
                st.success("🎉 Analysis Complete!")
                st.session_state.analysis_complete = True

                base_name = os.path.splitext(uploaded_file.name)[0]
                annotated_path = os.path.join(
                    PROJECT_ROOT,
                    "outputs",
                    f"annotated_{base_name}.mp4",
                )
                st.session_state.processed_video = (
                    annotated_path if os.path.exists(annotated_path) else None
                )

                if result.stdout.strip():
                    st.code(result.stdout.strip())
            else:
                st.session_state.analysis_complete = False
                error_message = result.stderr.strip() or result.stdout.strip()
                st.error(error_message or "Video analysis failed.")

st.divider()

# ==========================================================
# DASHBOARD
# ==========================================================

timeline_file = os.path.join(PROJECT_ROOT, "data", "timeline.csv")

if (
    uploaded_file is not None
    and st.session_state.analysis_complete
    and os.path.exists(timeline_file)
):
    dashboard = DashboardAnalytics(timeline_file)

    st.header("📊 Analysis Dashboard")

    # ----------------------------
    # Filters
    # ----------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_person = st.selectbox(
            "👤 Person",
            dashboard.person_list(),
        )

    with col2:
        selected_activity = st.selectbox(
            "🏃 Activity",
            dashboard.activity_list(),
        )

    with col3:
        search_query = st.text_input(
            "🔎 Search",
            placeholder="Search timestamp, person, or activity",
        )

    df = dashboard.filtered_data(
        selected_person,
        selected_activity,
        search_query,
    )

    st.divider()

    # ----------------------------
    # Metrics
    # ----------------------------

    m1, m2 = st.columns(2)

    with m1:
        st.metric(
            "👥 Total Persons",
            dashboard.total_persons(),
        )

    with m2:
        st.metric(
            "📋 Total Events",
            len(df),
        )

    st.divider()

    # ----------------------------
    # Timeline
    # ----------------------------

    st.subheader("📅 Activity Timelines")

    st.dataframe(
        df,
        use_container_width=True,
        height=450,
    )

    st.divider()

    # ----------------------------
    # Activity Chart
    # ----------------------------

    st.subheader("📈 Activity Distribution")

    counts = dashboard.activity_counts()

    if len(counts):
        st.bar_chart(counts)
    else:
        st.info("No activity data available.")

    st.divider()

    # ==========================================================
    # AI SUMMARY
    # ==========================================================

    st.subheader("🤖 AI Summary")

    summary = ActivitySummary(timeline_file)
    report = summary.generate()

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "👥 Total Persons",
            report["Total Persons"],
        )

        st.metric(
            "📋 Total Events",
            report["Total Events"],
        )

    with c2:
        st.metric(
            "🏃 Most Common Activity",
            report["Most Common Activity"],
        )

        st.metric(
            "⏱ Start Time",
            report["Start Time"],
        )

    c3, c4 = st.columns(2)

    with c3:
        st.metric(
            "⏱ End Time",
            report["End Time"],
        )

    with c4:
        st.write("**🚶 Movement Summary**")
        st.info(report["Movement Summary"])

    st.write("### 📊 Activity Counts")
    st.json(report["Activity Counts"])

    st.divider()

    # ==========================================================
    # ANNOTATED VIDEO
    # ==========================================================

    st.subheader("🎬 Annotated Output Video")

    processed_video = st.session_state.processed_video

    if processed_video and os.path.exists(processed_video):
        st.video(processed_video)
    else:
        st.info("Annotated video was not generated for this analysis run.")

    st.divider()

    # ==========================================================
    # DOWNLOAD
    # ==========================================================

    with open(timeline_file, "rb") as timeline_handle:
        timeline_bytes = timeline_handle.read()

    st.download_button(
        "📥 Download Timeline CSV",
        data=timeline_bytes,
        file_name="timeline.csv",
        mime="text/csv",
        use_container_width=True,
    )
