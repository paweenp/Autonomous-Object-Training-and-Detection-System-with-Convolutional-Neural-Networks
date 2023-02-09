import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
from PIL import Image
from helper_function import *

#img_path = "../data/cat/Cat_November_2010-1a.jpg"
img_path = "../data/otto/otto.5.jpg"
#img_path = "../data/flower_photos/sunflowers/3749091071_c146b33c74_n.jpg"
#img_path = "../data/flower_photos/dandelion/160456948_38c3817c6a_m.jpg"
#img_path = "../data/flower_photos/roses/909277823_e6fb8cb5c8_n.jpg"



image, img_raw = load_image_from_local(img_path, image_size=224, dynamic_size=False, max_dynamic_size=512 )

# Read labels from files
#labels_file = "labels/ImageNetLabels.txt"
labels_file = "labels.txt"
classes = []
with open(labels_file) as f:
  labels = f.readlines()
  classes = [l.strip() for l in labels]

# Load saved model
# trained_model_path = "train_model/train_model.h5"
# trained_model_path = "train_model/train_model_imagenet_resnet_v1_50_classification_5.h5"
trained_model_path = "train_model_imagenet_resnet_v1_50_classification_5.h5"

trained_model = tf.keras.models.load_model(trained_model_path, custom_objects={'KerasLayer':hub.KerasLayer})

predicted_scores = trained_model.predict(image)
predicted_index = np.argmax(predicted_scores)

print("predicted_scores : ", predicted_scores)
print("predicted_index : ", predicted_index)

prediction_class = classes[predicted_index]
prediction_percent = 10 * np.max(predicted_scores)

prediction_text  = prediction_class + " " + str(prediction_percent) + " %"

add_text_to_image(img_path, prediction_text)


