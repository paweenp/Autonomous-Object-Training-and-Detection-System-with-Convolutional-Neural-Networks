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
import matplotlib.pylab as plt
import datetime
import time
from helper_function import build_dataset
#from keras import layers
#from keras.applications.mobilenet_v2 import MobileNetV2

# 1. Build dataset from local files from folder
PATH = "../../../../Dataset/cats_and_dogs"
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_ds, val_ds, class_names = build_dataset(PATH, BATCH_SIZE, IMG_SIZE)
num_classes = len(class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# 2. Load pretrained model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                         include_top=False,
                         weights="imagenet")

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

# Freeze the model
base_model.trainable = False
feature_batch = base_model(image_batch)

# Add classification head to the model
global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_avg = global_avg_layer(feature_batch)
prediction_layer = tf.keras.layers.Dense(num_classes)
predicted_batch = prediction_layer(feature_batch_avg)

# Prepare the model
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_avg_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# 3. Train the model
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

predictions = model(image_batch)
predictions.shape

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.summary()
print("trainable variables : "+str(len(model.trainable_variables)))

initial_epochs = 10
loss0, accuracy0 = model.evaluate(val_ds)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds)

# 4. Evaluate the model performances

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


