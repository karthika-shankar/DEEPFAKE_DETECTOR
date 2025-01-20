import numpy as np
import pandas as pd
from keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm.notebook import tqdm_notebook as tqdm
import os

# Define paths
real = r"C:\c_programs\Side-Projects\test\dataset\Dataset\Train\Real"
fake = r"C:\c_programs\Side-Projects\test\dataset\Dataset\Train\Fake"

# Load image paths
real_path = os.listdir(real)
fake_path = os.listdir(fake)

def load_img(path):
    image = cv2.imread(path)
    if image is None:
        print(f"Error: Unable to load image at {path}")
        return None
    image = cv2.resize(image, (224, 224))  # Resize to 224x224
    return image[..., ::-1]  # Convert BGR to RGB

# Visualize real faces
fig = plt.figure(figsize=(10, 10))
for i in range(16):
    real_image_path = os.path.join(real, real_path[i])  # Correctly join paths
    plt.subplot(4, 4, i + 1)
    img = load_img(real_image_path)
    if img is not None:
        plt.imshow(img, cmap='gray')
    plt.suptitle("Real faces", fontsize=20)
    plt.axis('off')
plt.show()

fig = plt.figure(figsize=(10, 10))
for i in range(16):
    fake_image_path = os.path.join(fake, fake_path[i])  # Correctly join paths
    img = load_img(fake_image_path)
    if img is not None:
        plt.subplot(4, 4, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(fake_path[i][:4])  # Display the first 4 characters of the filename as title
    plt.suptitle("Fake faces", fontsize=20)
    plt.axis('off')
plt.show()

# Data augmentation
dataset_path_tr = r"C:\c_programs\Side-Projects\test\dataset\Dataset\Train"
dataset_path_t = r"C:\c_programs\Side-Projects\test\dataset\Dataset\Test"
dataset_path_v= r"C:\c_programs\Side-Projects\test\dataset\Dataset\Validation"

data_with_aug= ImageDataGenerator(horizontal_flip=True,
                                         vertical_flip=False,
                                         rescale=1./255)
train = data_with_aug.flow_from_directory(dataset_path_tr,
                                          class_mode="binary",
                                          target_size=(96, 96),
                                          batch_size=32,)
val = data_with_aug.flow_from_directory(dataset_path_v,
                                          class_mode="binary",
                                          target_size=(96, 96),
                                          batch_size=32,)

# MobileNetV2 model
mnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
tf.keras.backend.clear_session()

model = Sequential([mnet,
                    GlobalAveragePooling2D(),
                    Dense(512, activation="relu"),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(128, activation="relu"),
                    Dropout(0.1),
                    Dense(2, activation="softmax")])

model.layers[0].trainable = False

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

def scheduler(epoch):
    if epoch <= 2:
        return 0.001
    elif epoch > 2 and epoch <= 15:
        return 0.0001
    else:
        return 0.00001

lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)
hist = model.fit(train,
                 epochs=20,
                 callbacks=[lr_callbacks],
                 validation_data=val)

model.save('deepfake_detection_MNV2_model_2.h5')

# Visualizing accuracy and loss
epochs = 20
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.grid(True)
plt.legend(['Train', 'Validation'])
plt.style.use(['classic'])

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Validation Accuracy')
plt.grid(True)
plt.legend(['Train', 'Validation'], loc=4)
plt.style.use(['classic'])
plt.show()