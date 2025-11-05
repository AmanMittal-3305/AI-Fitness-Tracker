import os
import sys
import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
from aiortc.contrib.media import MediaRecorder

# ----------------------------------------------------------------------
# Set up imports
# ----------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

from process_frame import ProcessFrame
from utils import get_mediapipe_pose
from thresholds import get_thresholds_beginner, get_thresholds_pro

# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="AI Fitness Trainer", layout="wide")

st.title("🏋️‍♂️ AI Fitness Trainer (Live Camera)")

col1, col2 = st.columns(2)

with col1:
    mode = st.radio("Select Mode", ["Beginner", "Pro"], horizontal=True)
    if mode == "Beginner":
        thresholds = get_thresholds_beginner()
    else:
        thresholds = get_thresholds_pro()

with col2:
    exercise = st.selectbox(
        "Select Exercise",
        ["Squats", "Pushups", "Situps", "Bicep Curls"],
        index=0,
    )

st.markdown("---")

# ----------------------------------------------------------------------
# Initialize process frame and pose model
# ----------------------------------------------------------------------
pose = get_mediapipe_pose()
live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)

output_video_file = "output_live.flv"
if "download" not in st.session_state:
    st.session_state["download"] = False

# ----------------------------------------------------------------------
# WebRTC video processing
# ----------------------------------------------------------------------
def video_frame_callback(frame: av.VideoFrame):
    img = frame.to_ndarray(format="bgr24")

    # Process with selected exercise
    processed_frame, _ = live_process_frame.process(img, pose, exercise)

    return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")


def out_recorder_factory() -> MediaRecorder:
    """Record output video locally for download."""
    return MediaRecorder(output_video_file)


ctx = webrtc_streamer(
    key="ai-fitness-trainer",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=True),
    out_recorder_factory=out_recorder_factory,
)

# ----------------------------------------------------------------------
# Download recorded video
# ----------------------------------------------------------------------
download_button = st.empty()

if os.path.exists(output_video_file):
    with open(output_video_file, "rb") as op_vid:
        download = download_button.download_button(
            "⬇️ Download Recorded Session",
            data=op_vid,
            file_name="output_live.flv",
            mime="video/x-flv",
        )
        if download:
            st.session_state["download"] = True

# Remove old file after download
if os.path.exists(output_video_file) and st.session_state["download"]:
    os.remove(output_video_file)
    st.session_state["download"] = False
    download_button.empty()

# ----------------------------------------------------------------------
# Helpful tips
# ----------------------------------------------------------------------
st.markdown(
    """
### 📹 Tips:
- Ensure good lighting and full body visibility in the frame.  
- Keep the camera at waist or chest height for best accuracy.  
- You can switch exercises live without restarting the stream.
"""
)
