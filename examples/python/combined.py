# app_combined.py
import streamlit as st
from ultralytics import YOLO
from gtts import gTTS
from io import BytesIO
from PIL import Image
import cv2
import os
import tempfile
import yt_dlp
import streamlink

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

st.sidebar.subheader("2. Load YOLO Model")

# Define a default model path
DEFAULT_MODEL_PATH = "{{MODEL_URL}}"  # Placeholder for external program integration

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

st.title("🔍 YOLO Object Detection Suite")

st.markdown("""
Welcome to the **YOLO Object Detection Suite**! Select your preferred input method from the tabs below and perform real-time or batch detections using the YOLO model of your choice. Audio alerts will notify you of detected objects.
""")

# Define tabs for different input methods
tabs = st.tabs(["📷 Webcam", "▶️ YouTube", "📺 Twitch", "🖼️ Image Upload", "📁 Folder of Images", "🎬 Video Upload"])

# ----------------------------
# 1. Webcam Detection
# ----------------------------

with tabs[0]:
    st.header("📷 Webcam Detection")
    if model is not None:
        # Capture image from webcam
        captured_image = st.camera_input("Take a picture")

        if captured_image:
            image = Image.open(captured_image)
            st.image(image, caption='Captured Image.', use_column_width=True)

            # Perform detection
            results = model(image)
            annotated_image = results[0].plot()

            st.image(annotated_image, caption='Detected Image.', use_column_width=True)

            # Extract detections
            detections = results[0].boxes
            if len(detections) > 0:
                detected_classes = [model.names[int(box.cls)] for box in detections]
                unique_classes = list(set(detected_classes))
                detection_text = ", ".join([f"{cls} detected" for cls in unique_classes])
                st.success(detection_text)

                # Generate and play gTTS alert
                tts = gTTS(text=detection_text, lang='en')
                with BytesIO() as audio_buffer:
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)
                    st.audio(audio_buffer, format='audio/mp3')
            else:
                st.info("No detections made.")
    else:
        st.error("Please load a YOLO model to proceed.")

# ----------------------------
# 2. YouTube Detection
# ----------------------------

with tabs[1]:
    st.header("▶️ YouTube Video Detection")
    if model is not None:
        youtube_url = st.text_input("Enter YouTube Video URL:")

        if st.button("Process Video"):
            if youtube_url:
                try:
                    # Define yt-dlp options to download single-file format (MP4)
                    ydl_opts = {
                        'format': 'best[ext=mp4]/best',
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
                        frame_window = st.empty()
                        detection_summary = set()

                        with st.spinner("Processing video frames..."):
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

                                            # Generate and play gTTS alert
                                            tts = gTTS(text=detection_text, lang='en')
                                            with BytesIO() as audio_buffer:
                                                tts.write_to_fp(audio_buffer)
                                                audio_buffer.seek(0)
                                                st.audio(audio_buffer, format='audio/mp3')

                        cap.release()
                        os.remove(temp_video_path)
            else:
                st.error("Please enter a YouTube video URL.")
    else:
        st.error("Please load a YOLO model to proceed.")

# ----------------------------
# 3. Twitch Detection
# ----------------------------

with tabs[2]:
    st.header("📺 Twitch Stream Detection")
    if model is not None:
        twitch_url = st.text_input("Enter Twitch Stream URL:")

        run_stream = st.checkbox('Run Twitch Stream Detection')

        if run_stream and twitch_url:
            try:
                streams = streamlink.streams(twitch_url)
                if "best" not in streams:
                    st.error("No suitable stream found.")
                else:
                    stream = streams["best"].url
                    cap = cv2.VideoCapture(stream)
                    frame_window = st.empty()
                    detection_summary = set()

                    with st.spinner("Processing Twitch stream..."):
                        while cap.isOpened() and run_stream:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Failed to read from Twitch stream.")
                                break

                            # Convert frame to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(frame_rgb)

                            # Perform detection
                            results = model(pil_image)
                            annotated_frame = results[0].plot()

                            frame_window.image(annotated_frame, caption='Twitch Stream Frame', use_column_width=True)

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

                                        # Generate and play gTTS alert
                                        tts = gTTS(text=detection_text, lang='en')
                                        with BytesIO() as audio_buffer:
                                            tts.write_to_fp(audio_buffer)
                                            audio_buffer.seek(0)
                                            st.audio(audio_buffer, format='audio/mp3')

                    cap.release()
            except Exception as e:
                st.error(f"Error processing Twitch stream: {e}")
    else:
        st.error("Please load a YOLO model to proceed.")

# ----------------------------
# 4. Image Upload Detection
# ----------------------------

with tabs[3]:
    st.header("🖼️ Single Image Detection")
    if model is not None:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            # Perform detection
            results = model(image)
            annotated_image = results[0].plot()

            st.image(annotated_image, caption='Detected Image.', use_column_width=True)

            # Extract detections
            detections = results[0].boxes
            if len(detections) > 0:
                detected_classes = [model.names[int(box.cls)] for box in detections]
                unique_classes = list(set(detected_classes))
                detection_text = ", ".join([f"{cls} detected" for cls in unique_classes])
                st.success(detection_text)

                # Generate and play gTTS alert
                tts = gTTS(text=detection_text, lang='en')
                with BytesIO() as audio_buffer:
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)
                    st.audio(audio_buffer, format='audio/mp3')
            else:
                st.info("No detections made.")
    else:
        st.error("Please load a YOLO model to proceed.")

# ----------------------------
# 5. Folder of Images Detection
# ----------------------------

with tabs[4]:
    st.header("📁 Batch Image Detection")
    if model is not None:
        uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                st.image(image, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)

                # Perform detection
                results = model(image)
                annotated_image = results[0].plot()

                st.image(annotated_image, caption='Detected Image.', use_column_width=True)

                # Extract detections
                detections = results[0].boxes
                if len(detections) > 0:
                    detected_classes = [model.names[int(box.cls)] for box in detections]
                    unique_classes = list(set(detected_classes))
                    detection_text = ", ".join([f"{cls} detected" for cls in unique_classes])
                    st.success(f"{uploaded_file.name}: {detection_text}")

                    # Generate and play gTTS alert
                    tts = gTTS(text=detection_text, lang='en')
                    with BytesIO() as audio_buffer:
                        tts.write_to_fp(audio_buffer)
                        audio_buffer.seek(0)
                        st.audio(audio_buffer, format='audio/mp3')
                else:
                    st.info(f"{uploaded_file.name}: No detections made.")
    else:
        st.error("Please load a YOLO model to proceed.")

# ----------------------------
# 6. Video Upload Detection
# ----------------------------

with tabs[5]:
    st.header("🎬 Video File Detection")
    if model is not None:
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            # Save uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(uploaded_file.read())
                temp_video_path = temp_video.name

            cap = cv2.VideoCapture(temp_video_path)
            frame_window = st.empty()
            detection_summary = set()

            with st.spinner("Processing video frames..."):
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

                                # Generate and play gTTS alert
                                tts = gTTS(text=detection_text, lang='en')
                                with BytesIO() as audio_buffer:
                                    tts.write_to_fp(audio_buffer)
                                    audio_buffer.seek(0)
                                    st.audio(audio_buffer, format='audio/mp3')

            cap.release()
            os.remove(temp_video_path)
    else:
        st.error("Please load a YOLO model to proceed.")
