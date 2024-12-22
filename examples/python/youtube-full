import torch
from PIL import Image
import requests
from io import BytesIO
from pytube import YouTube
import cv2
import os

# ----------------------------
# User Input for Architecture Selection
# ----------------------------

print("Select YOLO architecture:")
print("1: YOLOv8")
print("2: YOLOv9")
print("3: YOLOv11")
print("4: YOLO-World")

architecture_choice = input("Enter the number corresponding to your architecture: ")

# Map user input to architecture
if architecture_choice == "1":
    architecture = "ultralytics/yolov8"
elif architecture_choice == "2":
    architecture = "ultralytics/yolov9"
elif architecture_choice == "3":
    architecture = "ultralytics/yolo11"
elif architecture_choice == "4":
    architecture = "ultralytics/yolo-world"
else:
    raise ValueError("Invalid architecture choice. Please select a number between 1 and 4.")

# ----------------------------
# Load the YOLO Model
# ----------------------------

# Placeholder for the model URL
MODEL_URL = "{{MODEL_URL}}"

# Load the model
model = torch.hub.load(architecture, 'custom', path_or_model=MODEL_URL)
model.eval()

# ----------------------------
# Function to Download YouTube Video
# ----------------------------

def download_youtube_video(video_url, output_path='youtube_video.mp4'):
    yt = YouTube(video_url)
    stream = yt.streams.filter(file_extension='mp4', res='720p').first()
    if not stream:
        raise ValueError("No suitable stream found. Please ensure the video has a 720p MP4 stream.")
    video_path = stream.download(output_path='.', filename=output_path)
    return video_path

# ----------------------------
# Perform Detection on YouTube Video
# ----------------------------

# Input YouTube video URL
video_url = input("Enter YouTube video URL: ")

# Download video
print("Downloading video...")
video_path = download_youtube_video(video_url)
print(f"Video downloaded to {video_path}")

# Open video for detection
cap = cv2.VideoCapture(video_path)

# Create output directory
output_dir = "youtube_output"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

print("Starting detection... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform inference
    results = model(img)

    # Render results on the frame
    results.render()
    detected_frame = results.imgs[0]
    detected_frame = cv2.cvtColor(detected_frame, cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2.imshow('YOLO YouTube Detection', detected_frame)

    # Save the frame to output directory
    output_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(output_path, detected_frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Detection completed. Processed {frame_count} frames. Results saved in '{output_dir}' directory.")
