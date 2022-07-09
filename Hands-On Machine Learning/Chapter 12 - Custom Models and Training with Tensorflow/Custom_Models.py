# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 07:26:21 2021

@author: Z52XXR7

Chapter 12 - Custom Models and Training with TensorFlow

"""

import tensorflow as tf
from tensorflow import keras

#Using TensorFlow like NumPy

tf.constant([[1,2,3],[4,5,6]])

tf.constant(42)

t = tf.constant([[1,2,3],[4,5,6]])
t.shape
t.dtype

t + 10

tf.square(t)


#Keras' Low-Level API

K = keras.backend
K.square(K.transpose(t)) + 10
