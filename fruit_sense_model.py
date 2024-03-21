import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers.experimental import RMSprop
from keras.layers import Dropout, Flatten, Dense

# Define data generators
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

img_height, img_width = 180, 180
batch_size = 6

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\Parth\Desktop\project\Banana Ripeness Classification.v2-original-images_modifiedclasses.folder\train",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\Parth\Desktop\project\Banana Ripeness Classification.v2-original-images_modifiedclasses.folder\valid",
    image_size=(img_height, img_width),  
    batch_size=batch_size
)

val_ds.class_names

class_names = train_ds.class_names



# Define and compile the model
resnet_model = Sequential()

pretrained_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(180, 180, 3),
    pooling='max',
    classes=4,
    weights='imagenet'
)

for layer in pretrained_model.layers:
    layer.trainable = False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(4, activation='softmax'))

resnet_model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Train the model
epochs = 12
history = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Plot accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Plot loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()



# Evaluate on test data
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\Parth\Desktop\project\Banana Ripeness Classification.v2-original-images_modifiedclasses.folder\test",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_loss, test_accuracy = resnet_model.evaluate(test_ds, verbose=1)
print("Loss  : ", test_loss)
print("Accuracy  :", test_accuracy)

