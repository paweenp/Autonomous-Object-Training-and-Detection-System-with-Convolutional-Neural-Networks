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
import os
from helper_function import build_dataset
#from keras import layers
#from keras.applications.mobilenet_v2 import MobileNetV2

# 1. Build dataset from local files from folder
#PATH = "../../../../Dataset/dataset_1"
PATH = "../../../../Dataset/cats_and_dogs"
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

#train_ds, val_ds, class_names = build_dataset(PATH, BATCH_SIZE, IMG_SIZE)
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
num_classes = len(class_names)

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)



#for image_batch, labels_batch in train_ds:
  #print(image_batch.shape)
  #print(labels_batch.shape)
  #break

# 2. Load pretrained model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                         include_top=False,
                         weights="imagenet")

base_model.summary()

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])


# Freeze the model
base_model.trainable = False
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print("prediction batch : "+str(feature_batch.shape))

# Add classification head to the model
global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_avg = global_avg_layer(feature_batch)

prediction_layer = tf.keras.layers.Dense(1)
#prediction_layer = tf.keras.layers.Dense(num_classes)
prediction_batch = prediction_layer(feature_batch_avg)
print("prediction batch : "+str(prediction_batch.shape))

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
              #loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()
print("trainable variables : "+str(len(model.trainable_variables)))

initial_epochs = 10
loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

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


