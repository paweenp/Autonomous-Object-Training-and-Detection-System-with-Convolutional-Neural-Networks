import tensorflow as tf
import tensorflow_hub as hub

import requests
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from helper_function import *

image_size = 224
dynamic_size = False

# Load saved model
model_path = "models/efficientnet_b0_classification_1"
model_image_size = 224
image_size = model_image_size

max_dynamic_size = 512
dynamic_size = False

# Path to label file
labels_file = "labels/ImageNetLabels.txt"

classes = []
with open(labels_file) as f:
  labels = f.readlines()
  classes = [l.strip() for l in labels]

img_path = "../data/cat/Cat_November_2010-1a.jpg"
#img_path = "../data/otto/otto.1.jpg"

image, original_image = load_image_from_local(
    img_path, 
    image_size, 
    dynamic_size, 
    max_dynamic_size)

show_image(image, 'Scaled image')

classifier = tf.saved_model.load(model_path)
#classifier = hub.load(model_path)

input_shape = image.shape
warmup_input = tf.random.uniform(input_shape, 0, 1.0)
# time warmup_logits = classifier(warmup_input).numpy()

# Run model on image 
probabilities = tf.nn.softmax(classifier(image)).numpy()

top_5 = tf.argsort(probabilities, axis=-1, direction="DESCENDING")[0][:5].numpy()
np_classes = np.array(classes)

# Some models include an additional 'background' class in the predictions, so
# we must account for this when reading the class labels.
includes_background_class = probabilities.shape[1] == 1001

for i, item in enumerate(top_5):
  class_index = item if includes_background_class else item + 1
  line = f'({i+1}) {class_index:4} - {classes[class_index]}: {probabilities[0][top_5][i]}'
  print(line)

show_image(image, '')