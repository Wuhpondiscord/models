# app_folder_upload.py
import streamlit as st
from ultralytics import YOLO
from gtts import gTTS
import os
from io import BytesIO
from PIL import Image

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

st.title("YOLO Batch Image Detection")

if model is not None:
    st.header("Upload Images for Detection")
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

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

                # Generate gTTS alert
                tts = gTTS(text=detection_text, lang='en')
                with BytesIO() as audio_buffer:
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)
                    st.audio(audio_buffer, format='audio/mp3')
            else:
                st.info(f"{uploaded_file.name}: No detections made.")
else:
    st.error("Please load a YOLO model to proceed.")
