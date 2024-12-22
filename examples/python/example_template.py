import torch
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

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

# Placeholder for the model URL or path
MODEL_URL = "{{MODEL_URL}}"

# Initialize the model based on the selected architecture
model = torch.hub.load(architecture, 'custom', path_or_model=MODEL_URL)

# Set the model to evaluation mode
model.eval()

# ----------------------------
# Perform Inference
# ----------------------------

# Replace the URL with the image you want to perform inference on
IMAGE_URL = 'https://ultralytics.com/images/zidane.jpg'

# Download the image
response = requests.get(IMAGE_URL)
img = Image.open(BytesIO(response.content))

# Perform inference
results = model(img)

# ----------------------------
# Visualize Results
# ----------------------------

# Display the image with bounding boxes
results.show()

# Alternatively, you can save the results
# results.save(save_dir='inference/output')
