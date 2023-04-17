import numpy as np
import os
import tensorflow as tf

def build_dataset(PATH, BATCH_SIZE, IMG_SIZE):
    
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')

    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)
    
    class_names = train_dataset.class_names
    class_names = np.array(train_dataset.class_names)
    
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    # tf.data methods used when load data
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = validation_dataset.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
    val_ds = validation_dataset.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.

    return train_ds, val_ds, class_names

def format_dataset_structure(PATH):
    return 0