import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
import detect_utils
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
from PIL import Image


input_path = 'inputs/cat/image_1.jpg'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_path = 'outputs'
detection_threshold = 0.8

# Read the image.
image = Image.open(input_path).convert('RGB')

# Create a BGR copy of the image for annotation.
image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# ------------- DETECTION -------------

#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')

# Load the model onto the computation device.
model = model.eval().to(device)

# Define the torchvision image transforms.
transform = transforms.Compose([transforms.ToTensor(),])

  # Transform the image to tensor.
image = transform(image).to(device)
# Add a batch dimension.
image = image.unsqueeze(0)

outputs = model(image)

# Get score for all the predicted objects.
pred_scores = outputs[0]['scores'].detach().cpu().numpy()
# Get all the predicted bounding boxes.
pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

# Get boxes above the threshold score.
boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
labels = outputs[0]['labels'][:len(boxes)]
# Get all the predicited class names.
pred_classes = [coco_names[i] for i in labels.cpu().numpy()]

# ------------- SHOW OUTPUTS -------------
# Draw bounding boxes.
image = detect_utils.draw_boxes(boxes, pred_classes, labels, image_bgr)

cv2.imshow('Image', image)
cv2.waitKey(0)