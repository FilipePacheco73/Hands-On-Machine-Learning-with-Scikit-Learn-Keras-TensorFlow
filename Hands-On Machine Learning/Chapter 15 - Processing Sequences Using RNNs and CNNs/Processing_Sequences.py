# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 09:43:40 2021

@author: Z52XXR7

Chapter 15 - Processing Sequences Using RNNs and CNNs

"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from matplotlib import pyplot as plt

#Forecasting a Time Series

def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = .5*np.sin((time-offsets1)*(freq1*10+10)) # wave 1
    series += .2*np.sin((time-offsets2)*(freq2*20+20)) # wave 2
    series += .1*(np.random.rand(batch_size, n_steps)-.5) # + noise
    return series[...,np.newaxis].astype(np.float32)

n_steps = 50
series = generate_time_series(10000, n_steps + 1)

#validating set

X_train, y_train = series[:7000,:n_steps], series[:7000,-1]
X_valid, y_valid = series[7000:9000,:n_steps], series[7000:9000,-1]
X_test, y_test = series[9000:,:n_steps], series[9000:,-1]


y_pred = X_valid[:,-1]
np.mean(keras.losses.mean_squared_error(y_valid,y_pred))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50,1]),
    keras.layers.Dense(1)
    ])

model = keras.models.Sequential([
    keras.layers.SimpleRNN(1,input_shape=[None,1])
    ])

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None,1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.SimpleRNN(1)
    ])

series = generate_time_series(1, n_steps+10)
X_new, y_new = series[:,:n_steps], series[:,n_steps:]
X = X_new
for step_ahead in range(10):
    y_pred_one = model.predict(X[:, step_ahead:])[:,np.newaxis,:]
    

