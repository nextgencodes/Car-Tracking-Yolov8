import torch
from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt")

# Load a 0-channel image
image = torch.rand(1, 3, 640, 640)

# Run the model
pred_labels = model(image)
print('predicted', pred_labels)