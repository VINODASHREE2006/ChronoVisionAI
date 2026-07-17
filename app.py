import os
import glob
import subprocess
import sys

import streamlit as st

from src.dashboard import DashboardAnalytics
from src.summary import ActivitySummary

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="ChronoVisionAI",
    page_icon="🎥",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("🎥 ChronoVisionAI")
st.subheader("AI Powered Human Activity Timeline Generator")

st.divider()

# =====================================================
# VIDEO UPLOAD
# =====================================================

st.header("📤 Upload CCTV Video")

uploaded_file = st.file_uploader(
    "Choose CCTV Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    os.makedirs("videos", exist_ok=True)

    with open("videos/test.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ Video Uploaded Successfully!")

    if st.button("🚀 Process Video"):

        with st.spinner("Running YOLOv8 + ByteTrack..."):

            result = subprocess.run(
                [sys.executable,"-m", "src.activity"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                st.success("✅ Processing Completed!")
            else:
                st.error(result.stderr)

st.divider()

# =====================================================
# MAIN LAYOUT
# =====================================================

left_col, right_col = st.columns([2, 1])
# =====================================================
# LEFT PANEL
# =====================================================

with left_col:

    st.header("🎥 Processed Video")

    processed_videos = sorted(
        glob.glob("runs/track/predict*/test.mp4")
    )

    if processed_videos:

        latest_video = processed_videos[-1]

        with open(latest_video, "rb") as video:
            st.video(video.read())

    elif os.path.exists("videos/test.mp4"):

        with open("videos/test.mp4", "rb") as video:
            st.video(video.read())

    else:

        st.info("Upload a video to begin.")

# =====================================================
# RIGHT PANEL
# =====================================================

with right_col:

    st.header("📋 Activity Timeline")

    if os.path.exists("data/timeline.csv"):
        analytics = DashboardAnalytics("data/timeline.csv")
        summary = ActivitySummary("data/timeline.csv")

        report = summary.generate()

        # ==========================================
        # FILTERS
        # ==========================================

        st.subheader("🔍 Search & Filter")

        person = st.selectbox(
            "Person ID",
            analytics.person_list()
        )

        activity = st.selectbox(
            "Activity",
            analytics.activity_list()
        )

        filtered_df = analytics.filtered_data(
            person,
            activity
        )

        # ==========================================
        # TIMELINE
        # ==========================================

        st.subheader("📝 Timeline")

        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=250
        )

        st.divider()

        # ==========================================
        # ANALYTICS
        # ==========================================

        st.subheader("📊 Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Events",
                len(filtered_df)
            )

        with col2:

            if analytics.person_col:

                persons = filtered_df[
                    analytics.person_col
                ].nunique()

            else:

                persons = 0

            st.metric(
                "Persons",
                persons
            )

        st.divider()

        # ==========================================
        # ACTIVITY DISTRIBUTION
        # ==========================================

        st.subheader("📈 Activity Distribution")

        if analytics.activity_col:

            chart = (
                filtered_df[
                    analytics.activity_col
                ]
                .value_counts()
            )

            st.bar_chart(chart)

        else:

            st.info("No activity data available.")

        st.divider()
        # ==========================================
        # AI SUMMARY
        # ==========================================

        st.subheader("🤖 AI Summary")

        st.info(f"""
### 📹 Video Summary

👥 **Total Persons:** {report['Total Persons']}

📋 **Total Events:** {report['Total Events']}

🚶 **Most Common Activity:** {report['Most Common Activity']}

🕒 **Start Time:** {report['Start Time']}

🕒 **End Time:** {report['End Time']}
""")

        st.divider()

        # ==========================================
        # DOWNLOAD CSV
        # ==========================================

        st.download_button(
            label="📥 Download Timeline CSV",
            data=filtered_df.to_csv(index=False),
            file_name="timeline.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "⚠️ No timeline found. Upload and process a video first."
        )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;padding:10px;">
        <h3>🎥 ChronoVisionAI</h3>
        <p><b>AI Powered Human Activity Timeline Generator</b></p>
        <p>YOLOv8 • ByteTrack • OpenCV • Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)