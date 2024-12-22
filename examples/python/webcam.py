import torch
from PIL import Image
import cv2

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
# Perform Detection on Webcam Feed
# ----------------------------

# Open webcam feed
cap = cv2.VideoCapture(0)

print("Starting webcam detection. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
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
    cv2.imshow('YOLO Webcam Detection', detected_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Webcam detection ended.")
