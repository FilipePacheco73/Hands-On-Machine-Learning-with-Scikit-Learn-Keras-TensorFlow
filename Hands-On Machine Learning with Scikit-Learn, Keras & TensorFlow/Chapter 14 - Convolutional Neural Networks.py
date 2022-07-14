# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 07:26:21 2021

@author: Filipe Pacheco

Hands-On Machine Learning

Chapter 14 - Deep Computer Vision Using Convulutional Neural Networks

"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

#TensorFlow Implementation

from sklearn.datasets import load_sample_image 



#Load sample images

china = load_sample_image("china.jpg")/255
flower = load_sample_image("flower.jpg")/255

images = np.array([china,flower])
batch_size, height, width, channels = images.shape



#Create 2 filters

filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:,3,:,0] = 1 # vertical line
filters[3,:,:,1] = 1 # horizontal line

outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")

plt.imshow(china,cmap="gray")
plt.imshow(outputs[0,:,:,1],cmap="gray")

conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                           padding="same", activation="relu")

#Creating a Max Pooling

max_pool = keras.layers.MaxPool2D(pool_size=2)

model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation="relu", padding="same",input_shape=[28,28,1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
    ])