import torch
from PIL import Image
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
# Perform Detection on Folder of Images
# ----------------------------

# Define folders
image_folder = "./images"
output_folder = "./output"
os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
supported_extensions = ('.png', '.jpg', '.jpeg')

# Process each image in the folder
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith(supported_extensions):
        img_path = os.path.join(image_folder, image_file)
        img = Image.open(img_path)

        # Perform inference
        results = model(img)

        # Save results
        results.save(save_dir=output_folder)
        print(f"Processed {image_file}")

print(f"All images processed. Results saved in '{output_folder}' directory.")
