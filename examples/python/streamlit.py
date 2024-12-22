import torch
from PIL import Image
import cv2
import streamlit as st
import streamlink
import tempfile
import time
import os
from pytube import YouTube
import requests
from io import BytesIO
from ultralytics import YOLO
import numpy as np
from gtts import gTTS
import zipfile

# Set Streamlit page configuration
st.set_page_config(
    page_title="🔍 YOLO Object Detection App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# User Input for Architecture Selection
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
DEFAULT_MODEL_PATH = "{{MODEL_URL}}"

model = None
model_loaded = False

if os.path.exists(DEFAULT_MODEL_PATH):
    try:
        model = YOLO(DEFAULT_MODEL_PATH)
        model_loaded = True
        st.sidebar.success(f"✅ Model loaded successfully from '{DEFAULT_MODEL_PATH}'.")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model from '{DEFAULT_MODEL_PATH}': {e}")
else:
    st.sidebar.warning(f"⚠️ Default model '{DEFAULT_MODEL_PATH}' not found.")

if not model_loaded:
    st.sidebar.info("🔄 Please upload a YOLO model or specify a model path.")

    uploaded_model = st.sidebar.file_uploader(
        "Upload your YOLO .pt model file",
        type=["pt"],
        help="Upload a trained YOLO model file (.pt)."
    )

    model_path_input = st.sidebar.text_input(
        "Or Specify Model Path",
        value="",
        help="Enter the path to your YOLO model file if not uploading."
    )

    if uploaded_model is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
            tmp_model.write(uploaded_model.read())
            tmp_model_path = tmp_model.name
        try:
            model = YOLO(tmp_model_path)
            model_loaded = True
            st.sidebar.success("✅ YOLO model uploaded and loaded successfully.")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading uploaded model: {e}")
        finally:
            # Clean up the temporary file
            os.unlink(tmp_model_path)
    elif model_path_input:
        if os.path.exists(model_path_input):
            try:
                model = YOLO(model_path_input)
                model_loaded = True
                st.sidebar.success("✅ YOLO model loaded successfully from the specified path.")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading model from path: {e}")
        else:
            st.sidebar.error(f"❌ Specified model path '{model_path_input}' does not exist.")

# ----------------------------
# TTS Configuration
# ----------------------------

st.sidebar.subheader("3. Text-to-Speech (TTS) Settings")

tts_enabled = st.sidebar.checkbox("Enable Detection Announcements", value=True)

voice_options = {
    "English (US)": "en",
    "English (UK)": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}

selected_voice = st.sidebar.selectbox(
    "Select Voice/Accent for Announcements:",
    options=list(voice_options.keys()),
)

# ----------------------------
# Function to Generate TTS Audio
# ----------------------------

def generate_tts(message, lang):
    tts = gTTS(text=message, lang=lang, slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
        tts.save(tmp_audio.name)
        return tmp_audio.name

# ----------------------------
# Function to Perform Object Detection
# ----------------------------

def detect_objects(img, model, confidence_thresholds, enabled_classes):
    # Run inference
    results = model(img)

    annotated_img = img.copy()
    detected_classes = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, "Unknown")
            conf = box.conf[0].item()

            if cls_name in confidence_thresholds:
                threshold = confidence_thresholds[cls_name]
                if conf < threshold or cls_name not in enabled_classes:
                    continue

                detected_classes.append(cls_name)

                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = f"{cls_name} {conf:.2f}"
                # Draw rectangle and label on the image
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    return annotated_img, detected_classes

# ----------------------------
# Function to Get Twitch Stream URL
# ----------------------------

def get_twitch_stream_url(channel_name, quality="best"):
    try:
        # Use streamlink to get the stream URL
        streams = streamlink.streams(f"twitch.tv/{channel_name}")
        if not streams:
            st.error("No streams found for the specified channel.")
            return None
        stream = streams.get(quality)
        if not stream:
            st.error(f"Quality '{quality}' not available.")
            return None
        return stream.url
    except Exception as e:
        st.error(f"Error obtaining stream URL: {e}")
        return None

# ----------------------------
# Function to Download YouTube Video
# ----------------------------

def download_youtube_video(youtube_url):
    try:
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if not stream:
            st.error("No suitable streams found.")
            return None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            stream.download(output_path=os.path.dirname(tmp_video.name), filename=os.path.basename(tmp_video.name))
            return tmp_video.name
    except Exception as e:
        st.error(f"Error downloading YouTube video: {e}")
        return None

# ----------------------------
# Function to Fetch Image from URL
# ----------------------------

def fetch_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(img)
    except Exception as e:
        st.error(f"Error fetching image from URL: {e}")
        return None

# ----------------------------
# Function to Handle Folder Upload (Zip File)
# ----------------------------

def extract_zip(zip_file):
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
            return tmp_dir
    except Exception as e:
        st.error(f"Error extracting zip file: {e}")
        return None

# ----------------------------
# Streamlit App
# ----------------------------

def main():
    st.title("🔍 YOLO Object Detection App with TTS")
    st.markdown("""
    This application allows you to perform object detection using your YOLO model on various input sources, including YouTube videos, Twitch streams, uploaded files, folder uploads, and image URLs. Additionally, it can announce detections with customizable voices and accents.
    """)

    if not model_loaded:
        st.warning("⚠️ Please load a YOLO model to proceed.")
        return

    # After model is loaded, get class names
    class_names = model.names
    if not class_names:
        st.error("❌ No class names found in the model.")
        return

    # ----------------------------
    # Per-Class Configuration
    # ----------------------------

    st.sidebar.subheader("4. Per-Class Configuration")
    st.sidebar.markdown("Adjust the confidence threshold for each class and toggle classes on/off.")

    confidence_thresholds = {}
    enabled_classes = set()

    for cls_id, cls in class_names.items():
        threshold = st.sidebar.slider(
            f"Confidence Threshold for {cls}",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            key=f"threshold_{cls}"
        )
        confidence_thresholds[cls] = threshold

    # Class toggles
    for cls_id, cls in class_names.items():
        enabled = st.sidebar.checkbox(f"Enable {cls}", value=True, key=f"toggle_{cls}")
        if enabled:
            enabled_classes.add(cls)

    # ----------------------------
    # Select Input Method
    # ----------------------------

    st.sidebar.subheader("5. Select Input Method")
    input_method = st.sidebar.selectbox(
        "Choose Input Method",
        options=["YouTube Video", "Twitch Stream", "Upload Files", "Folder Upload", "Image URL"],
        help="Select the method you want to use to test the model."
    )

    # Initialize variables
    detected_messages = []
    audio_files = []

    # ----------------------------
    # Handle Different Input Methods
    # ----------------------------

    if input_method == "YouTube Video":
        st.header("📹 YouTube Video Object Detection")

        youtube_url = st.text_input(
            "YouTube Video URL",
            value="https://www.youtube.com/watch?v=your_video_id",
            help="Enter the full URL of the YouTube video you want to analyze."
        )

        if st.button("Start YouTube Video Detection"):
            if not youtube_url:
                st.error("Please enter a YouTube video URL.")
            else:
                video_path = download_youtube_video(youtube_url)
                if video_path:
                    st.success("✅ YouTube video downloaded successfully.")

                    cap = cv2.VideoCapture(video_path)

                    if not cap.isOpened():
                        st.error("❌ Unable to open the video file.")
                    else:
                        frame_placeholder = st.empty()

                        st.info("🔄 Processing the video. This may take a moment...")
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                st.warning("⚠️ Reached the end of the video.")
                                break

                            # Perform object detection
                            annotated_frame, detected_classes = detect_objects(frame, model, confidence_thresholds, enabled_classes)

                            # Generate detection messages
                            for cls in detected_classes:
                                message = f"{cls} detected"
                                detected_messages.append(message)
                                if tts_enabled:
                                    lang = voice_options.get(selected_voice, "en")
                                    audio_path = generate_tts(message, lang)
                                    audio_files.append(audio_path)

                            # Convert to RGB and display
                            annotated_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                            frame_placeholder.image(annotated_image, caption="🔍 Detection Results", use_column_width=True)

                            # Display detection messages
                            for msg in detected_messages:
                                st.write(f"✅ {msg}")

                            # Play TTS audio
                            for audio in audio_files:
                                st.audio(audio, format="audio/mp3")
                                os.remove(audio)
                            audio_files = []

                            detected_messages = []

                            # Control frame rate
                            time.sleep(0.03)  # Approximately 30 FPS

                        cap.release()
                        # Optionally, delete the downloaded video
                        os.remove(video_path)

    elif input_method == "Twitch Stream":
        st.header("🔴 Twitch Stream Object Detection")

        twitch_channel = st.text_input(
            "Twitch Channel Name",
            value="your_channel_name",
            help="Enter the Twitch channel name you want to analyze."
        )
        quality = st.selectbox(
            "Stream Quality",
            options=["best", "720p", "480p", "360p"],
            index=0,
            help="Select the quality of the Twitch stream."
        )

        if st.button("Start Twitch Stream Detection"):
            if not twitch_channel:
                st.error("Please enter a Twitch channel name.")
            else:
                stream_url = get_twitch_stream_url(twitch_channel, quality)
                if stream_url:
                    st.success(f"✅ Successfully obtained stream URL for channel: **{twitch_channel}**")

                    # Initialize video capture
                    cap = cv2.VideoCapture(stream_url)

                    if not cap.isOpened():
                        st.error("❌ Unable to open the Twitch stream. Please check the channel name and ensure the stream is live.")
                    else:
                        frame_placeholder = st.empty()

                        st.info("🔄 Processing the Twitch stream. This may take a moment...")
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                st.warning("⚠️ Unable to read frame from the stream. Stream may have ended.")
                                break

                            # Perform object detection
                            annotated_frame, detected_classes = detect_objects(frame, model, confidence_thresholds, enabled_classes)

                            # Generate detection messages
                            for cls in detected_classes:
                                message = f"{cls} detected"
                                detected_messages.append(message)
                                if tts_enabled:
                                    lang = voice_options.get(selected_voice, "en")
                                    audio_path = generate_tts(message, lang)
                                    audio_files.append(audio_path)

                            # Convert to RGB and display
                            annotated_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                            frame_placeholder.image(annotated_image, caption="🔍 Detection Results", use_column_width=True)

                            # Display detection messages
                            for msg in detected_messages:
                                st.write(f"✅ {msg}")

                            # Play TTS audio
                            for audio in audio_files:
                                st.audio(audio, format="audio/mp3")
                                os.remove(audio)
                            audio_files = []

                            detected_messages = []

                            # Control frame rate
                            time.sleep(0.03)  # Approximately 30 FPS

                        cap.release()

    elif input_method == "Upload Files":
        st.header("📂 Upload Images or Videos for Object Detection")

        uploaded_files = st.file_uploader(
            "Upload Image or Video Files",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
            accept_multiple_files=True,
            help="Upload one or more image/video files."
        )

        if st.button("Start File Detection"):
            if not uploaded_files:
                st.error("Please upload at least one file.")
            else:
                for uploaded_file in uploaded_files:
                    st.write(f"### Processing: {uploaded_file.name}")

                    # Determine file type
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

                    if file_extension in [".jpg", ".jpeg", ".png"]:
                        # Process image
                        try:
                            image = Image.open(uploaded_file).convert("RGB")
                            img_array = np.array(image)
                            annotated_img, detected_classes = detect_objects(img_array, model, confidence_thresholds, enabled_classes)
                            annotated_image = Image.fromarray(annotated_img)
                            st.image(annotated_image, caption="🔍 Detection Results", use_column_width=True)

                            # Generate detection messages
                            for cls in detected_classes:
                                message = f"{cls} detected"
                                detected_messages.append(message)
                                if tts_enabled:
                                    lang = voice_options.get(selected_voice, "en")
                                    audio_path = generate_tts(message, lang)
                                    audio_files.append(audio_path)

                            # Display detection messages
                            for msg in detected_messages:
                                st.write(f"✅ {msg}")

                            # Play TTS audio
                            for audio in audio_files:
                                st.audio(audio, format="audio/mp3")
                                os.remove(audio)
                            audio_files = []

                            detected_messages = []

                        except Exception as e:
                            st.error(f"❌ Error processing image: {e}")

                    elif file_extension in [".mp4", ".avi", ".mov"]:
                        # Save uploaded video to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_video:
                            tmp_video.write(uploaded_file.read())
                            tmp_video_path = tmp_video.name

                        cap = cv2.VideoCapture(tmp_video_path)

                        if not cap.isOpened():
                            st.error(f"❌ Unable to open the video file: {uploaded_file.name}")
                        else:
                            frame_placeholder = st.empty()

                            st.info(f"🔄 Processing video: {uploaded_file.name}")
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret:
                                    st.warning("⚠️ Reached the end of the video.")
                                    break

                                # Perform object detection
                                annotated_frame, detected_classes = detect_objects(frame, model, confidence_thresholds, enabled_classes)

                                # Generate detection messages
                                for cls in detected_classes:
                                    message = f"{cls} detected"
                                    detected_messages.append(message)
                                    if tts_enabled:
                                        lang = voice_options.get(selected_voice, "en")
                                        audio_path = generate_tts(message, lang)
                                        audio_files.append(audio_path)

                                # Convert to RGB and display
                                annotated_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                                frame_placeholder.image(annotated_image, caption="🔍 Detection Results", use_column_width=True)

                                # Display detection messages
                                for msg in detected_messages:
                                    st.write(f"✅ {msg}")

                                # Play TTS audio
                                for audio in audio_files:
                                    st.audio(audio, format="audio/mp3")
                                    os.remove(audio)
                                audio_files = []

                                detected_messages = []

                                # Control frame rate
                                time.sleep(0.03)  # Approximately 30 FPS

                            cap.release()
                            # Optionally, delete the temporary video file
                            os.remove(tmp_video_path)

    elif input_method == "Folder Upload":
        st.header("📁 Upload a Folder (Zip) for Object Detection")

        uploaded_zip = st.file_uploader(
            "Upload a Zip File Containing Images/Videos",
            type=["zip"],
            help="Upload a zip file containing image and/or video files."
        )

        if st.button("Start Folder Detection"):
            if not uploaded_zip:
                st.error("Please upload a zip file.")
            else:
                extracted_dir = extract_zip(uploaded_zip)
                if extracted_dir:
                    st.success("✅ Zip file extracted successfully.")
                    # List all files in the extracted directory
                    for root, dirs, files in os.walk(extracted_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            st.write(f"### Processing: {file}")

                            file_extension = os.path.splitext(file)[1].lower()

                            if file_extension in [".jpg", ".jpeg", ".png"]:
                                # Process image
                                try:
                                    image = Image.open(file_path).convert("RGB")
                                    img_array = np.array(image)
                                    annotated_img, detected_classes = detect_objects(img_array, model, confidence_thresholds, enabled_classes)
                                    annotated_image = Image.fromarray(annotated_img)
                                    st.image(annotated_image, caption="🔍 Detection Results", use_column_width=True)

                                    # Generate detection messages
                                    for cls in detected_classes:
                                        message = f"{cls} detected"
                                        detected_messages.append(message)
                                        if tts_enabled:
                                            lang = voice_options.get(selected_voice, "en")
                                            audio_path = generate_tts(message, lang)
                                            audio_files.append(audio_path)

                                    # Display detection messages
                                    for msg in detected_messages:
                                        st.write(f"✅ {msg}")

                                    # Play TTS audio
                                    for audio in audio_files:
                                        st.audio(audio, format="audio/mp3")
                                        os.remove(audio)
                                    audio_files = []

                                    detected_messages = []

                                except Exception as e:
                                    st.error(f"❌ Error processing image: {e}")

                            elif file_extension in [".mp4", ".avi", ".mov"]:
                                # Process video
                                cap = cv2.VideoCapture(file_path)

                                if not cap.isOpened():
                                    st.error(f"❌ Unable to open the video file: {file}")
                                else:
                                    frame_placeholder = st.empty()

                                    st.info(f"🔄 Processing video: {file}")
             
