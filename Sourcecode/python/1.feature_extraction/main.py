# --------------------------------------------
# 1. Feature-Extraction Transfer-learning
# --------------------------------------------

# main program will perform the following
# 0. Import libraries
# 1. Build dataset from local files from folder
# 2. Load pretrained model
# 3. Train the model
# 4. Evaluate the performance
# --------------------------------------------

# 0. Import libraries
import tensorflow as tf
import tensorflow_hub as hub
import os
import matplotlib.pylab as plt
import datetime
import time
import numpy as np

# 1. Build dataset from local files from folder
PATH = "../../../"
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

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
num_classes = len(class_names)

# show_input = False
# 
# if show_input == True:
#     plt.figure(figsize=(10, 10))
#     for images, labels in train_dataset.take(1):
#      for i in range(9):
#          ax = plt.subplot(3, 3, i + 1)
#          plt.imshow(images[i].numpy().astype("uint8"))
#          plt.title(class_names[labels[i]])
#          plt.axis("off")
#     plt.show()

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = validation_dataset.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
val_ds = validation_dataset.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.

# tf.data methods used when load data
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


# 2. Load pretrained model
mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4" # headless model

feature_extractor_model = mobilenet_v2

feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model,
    input_shape=(224, 224, 3),
    trainable=False)

feature_batch = feature_extractor_layer(image_batch)
#print(feature_batch.shape)

model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(num_classes)
])

model.summary()

# 3. Train the model

predictions = model(image_batch)
predictions.shape

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1) # Enable histogram computation for every epoch.

NUM_EPOCHS = 10

model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS)

predicted_batch = model.predict(image_batch)
predicted_id = tf.math.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]  
true_label_batch = class_names[labels_batch]


# 4. Evaluate the model performances

# .... (Work in progress) ...


# Save model
t = time.time()

export_path = "/saved_models/{}".format(int(t))
export_path = "saved_models/saved_model"
model.save(export_path)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)

for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  txt = "True: "+true_label_batch[n]+" \n Predicted: "+predicted_label_batch[n].title()
  plt.title(txt)
  plt.axis('off')
_ = plt.suptitle("Model predictions")
plt.show()


