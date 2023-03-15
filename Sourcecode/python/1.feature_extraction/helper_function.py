import numpy as np
import cv2
import os
import tensorflow as tf

def build_dataset(subset, data_dir, IMAGE_SIZE, split_ratio):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=split_ratio,
        subset=subset,
        label_mode="categorical",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=1)
    return dataset

def modify_model(base_model, IMG_DIM):
    
    model = tf.keras.Sequential([
		tf.keras.layers.InputLayer(input_shape=IMG_DIM + (3,)),
  
	])