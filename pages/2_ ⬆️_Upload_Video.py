import os
import sys
import tempfile
import cv2
import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner, get_thresholds_pro

st.set_page_config(layout="wide")
st.title('AI Fitness Trainer — Video Analysis (Pushups / Squats / Situps / Bicep Curls)')

mode = st.radio('Select Mode', ['Beginner', 'Pro'], horizontal=True)
exercise = st.selectbox('Select Exercise', ['Push-ups', 'Squats', 'Sit-ups', 'Bicep Curls'])

thresholds = get_thresholds_beginner() if mode == 'Beginner' else get_thresholds_pro()
processor = ProcessFrame(thresholds=thresholds, flip_frame=False)

pose = get_mediapipe_pose()

if 'download' not in st.session_state:
    st.session_state['download'] = False

output_video_file = 'output_recorded.mp4'
if os.path.exists(output_video_file):
    os.remove(output_video_file)

with st.form('Upload', clear_on_submit=True):
    up_file = st.file_uploader("Upload a Video (mp4, mov, avi)", ['mp4', 'mov', 'avi'])
    uploaded = st.form_submit_button("Analyze Video")

stframe = st.empty()
sidebar_video = st.sidebar.empty()
warn = st.empty()
download_button = st.empty()

if up_file and uploaded:
    warn.empty()
    if os.path.exists(output_video_file):
        os.remove(output_video_file)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    try:
        tfile.write(up_file.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)

        # Video writer setup (match input fps & size)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

        sidebar_video.video(tfile.name)

        # Process frames
        frame_idx = 0
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Process returns BGR frame with drawings
            out_frame_bgr, metadata = processor.process(frame_bgr, pose, exercise)

            # show in streamlit (convert BGR->RGB for display)
            stframe.image(cv2.cvtColor(out_frame_bgr, cv2.COLOR_BGR2RGB), channels="RGB")

            # write to output video (BGR)
            video_output.write(out_frame_bgr)

            frame_idx += 1

        cap.release()
        video_output.release()
        tfile.close()

    except Exception as e:
        warn.markdown(f"**Error processing video:** {e}")

# Allow download when file exists
if os.path.exists(output_video_file):
    with open(output_video_file, 'rb') as op_vid:
        download = download_button.download_button('Download Analysis Video', data=op_vid, file_name=output_video_file)
    if download:
        st.session_state['download'] = True

if os.path.exists(output_video_file) and st.session_state.get('download', False):
    os.remove(output_video_file)
    st.session_state['download'] = False
    download_button.empty()
