import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt

# ----------------------------
# Load the YOLO Model
# ----------------------------
# Replace the placeholder with the actual model URL
MODEL_URL = "{{MODEL_URL}}"

# Download the model weights
model_weights = torch.hub.load_state_dict_from_url(MODEL_URL, map_location='cpu')

# Initialize the model (ensure the architecture matches your trained model)
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=MODEL_URL)

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
