import tensorflow as tf

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import matplotlib.pyplot as plt
import numpy as np


original_image_cache = {}

def preprocess_image(image):
  
  image = np.array(image)
  # reshape into shape [batch_size, height, width, num_channels]
  img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
  return image

def load_image_from_local(img_path, image_size=256, dynamic_size=False, max_dynamic_size=512 ):
    """Returns an image with shape [1, height, width, num_channels]."""
    img = Image.open(img_path)
    img = preprocess_image(img)
	
 	# Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img_raw = img	
    if tf.reduce_max(img) > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    if not dynamic_size:
        img = tf.image.resize_with_pad(img, image_size, image_size)
    elif img.shape[1] > max_dynamic_size or img.shape[2] > max_dynamic_size:
        img = tf.image.resize_with_pad(img, max_dynamic_size, max_dynamic_size)
    return img, img_raw

def show_image(image, title=''):
  image_size = image.shape[1]
  w = (image_size * 6) // 320
  plt.figure(figsize=(w, w))
  plt.imshow(image[0], aspect='equal')
  plt.axis('off')
  plt.title(title)
  plt.show()
  
def build_dataset(subset, data_dir, IMAGE_SIZE):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.20,
        subset=subset,
        label_mode="categorical",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=1)
    return dataset

def add_text_to_image(img_path, txt):
    image = Image.open(img_path)
    # Call draw Method to add 2D graphics in an image
    I1 = ImageDraw.Draw(image)
    
    fnt_size = int(0.1 * image.size[0])
    fnt = ImageFont.truetype("arial.ttf", fnt_size)
    I1.text((10, 10), txt, font=fnt, fill =(255, 0, 0))
    image.show()