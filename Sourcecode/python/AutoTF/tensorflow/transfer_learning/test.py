import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
from PIL import Image
from helper_function import *

img_path = "../data/otto/otto.5.jpg"

image, img_raw = load_image_from_local(img_path, image_size=224, dynamic_size=False, max_dynamic_size=512 )


# Load saved model
trained_model_path = "train_model.h5"

trained_model = tf.keras.models.load_model(trained_model_path, custom_objects={'KerasLayer':hub.KerasLayer})

predicted_scores = trained_model.predict(image)
predicted_index = np.argmax(predicted_scores)


print("predicted_scores : ", predicted_scores)
print("predicted_index : ", predicted_index)