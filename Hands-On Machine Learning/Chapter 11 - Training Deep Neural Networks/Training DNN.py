# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 10:49:00 2021

@author: filip

Chapter 11 - Training Deep Neural Networks

"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

#Glorot & He Initilization

keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")

he_avg_init = keras.initializers.VarianceScaling(scale = 2, mode="fan_avg", distribution="uniform")

keras.layers.Dense(0, activation="sigmoid", kernel_initializer=he_avg_init)

#Batch Normalization

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
    ])

model.summary()
