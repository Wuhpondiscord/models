# app_youtube.py
import streamlit as st
from ultralytics import YOLO
from gtts import gTTS
import os
from io import BytesIO
from PIL import Image
import cv2
import tempfile
import yt_dlp

# ----------------------------
# Configuration Sidebar
# ----------------------------

st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("1. Select YOLO Architecture")
architecture_choice = st.sidebar.selectbox(
    "Select YOLO architecture:",
    options=["YOLOv8", "YOLOv9", "YOLOv11", "YOLO-World"],
)

# Map user input to architecture
architecture_map = {
    "YOLOv8": "ultralytics/yolov8",
    "YOLOv9": "ultralytics/yolov9",
    "YOLOv11": "ultralytics/yolo11",
    "YOLO-World": "ultralytics/yolo-world",
}

architecture = architecture_map.get(architecture_choice)

# ----------------------------
# Model Auto-Load or Upload
# ----------------------------

st.sidebar.subheader("2. Load YOLO Model")

# Define a default model path
DEFAULT_MODEL_PATH = "{{MODEL_URL}}"  # Replace with your default model path

model_path = st.sidebar.text_input(
    "Model Path or URL",
    value=DEFAULT_MODEL_PATH,
    help="Enter the path or URL to the YOLO model."
)

model = None
if model_path and model_path != "{{MODEL_URL}}":
    try:
        model = YOLO(model_path)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
elif model_path == "{{MODEL_URL}}":
    st.sidebar.info("Model will be loaded by the external program.")

# ----------------------------
# Main Application
# ----------------------------

st.title("YOLO YouTube Video Detection")

if model is not None:
    st.header("Process a YouTube Video for Detection")
    youtube_url = st.text_input("Enter YouTube Video URL:")

    if st.button("Process Video"):
        if youtube_url:
            try:
                # Define yt-dlp options to download single-file format (MP4)
                ydl_opts = {
                    'format': 'best[ext=mp4]/best',  # Prioritize MP4 single-file formats
                    'outtmpl': tempfile.gettempdir() + '/%(id)s.%(ext)s',
                    'quiet': True,
                    'no_warnings': True,
                }

                with st.spinner("Downloading video..."):
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info_dict = ydl.extract_info(youtube_url, download=True)
                        video_id = info_dict.get('id', None)
                        video_ext = info_dict.get('ext', None)
                        video_filename = f"{video_id}.{video_ext}"
                        temp_video_path = os.path.join(tempfile.gettempdir(), video_filename)

                if not os.path.exists(temp_video_path):
                    st.error("Failed to download the video.")
                else:
                    cap = cv2.VideoCapture(temp_video_path)
                    frame_window = st.image([])
                    detection_summary = set()

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Convert frame to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)

                        # Perform detection
                        results = model(pil_image)
                        annotated_frame = results[0].plot()

                        frame_window.image(annotated_frame, caption='Video Frame', use_column_width=True)

                        # Extract detections
                        detections = results[0].boxes
                        if len(detections) > 0:
                            detected_classes = [model.names[int(box.cls)] for box in detections]
                            unique_classes = list(set(detected_classes))
                            for cls in unique_classes:
                                if cls not in detection_summary:
                                    detection_summary.add(cls)
                                    detection_text = f"{cls} detected"
                                    st.success(detection_text)

                                    # Generate gTTS alert
                                    tts = gTTS(text=detection_text, lang='en')
                                    with BytesIO() as audio_buffer:
                                        tts.write_to_fp(audio_buffer)
                                        audio_buffer.seek(0)
                                        st.audio(audio_buffer, format='audio/mp3')

                    cap.release()
                    os.remove(temp_video_path)
            except yt_dlp.utils.DownloadError as de:
                st.error(f"yt-dlp Download Error: {de}")
            except Exception as e:
                st.error(f"Error processing YouTube video: {e}")
        else:
            st.error("Please enter a YouTube video URL.")
else:
    st.error("Please load a YOLO model to proceed.")
