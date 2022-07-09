# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:16:30 2021

@author: Z52XXR7

Chapter 16 - Natural Language Processing with RNNs and Attention

"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from matplotlib import pyplot as plt

shakespeare_url = "https://homl.info/shakespeare"

filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)

with open(filepath) as f:
    shakespeare_text = f.read()
    
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(shakespeare_text)

tokenizer.texts_to_sequences(["First"])

tokenizer.sequences_to_texts([[20,6,9,8,3]])

max_id = len(tokenizer.word_index) # number of distinct characters
dataset_size = tokenizer.document_count # total number of characters

[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1

train_size = dataset_size*90//100

dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

n_steps = 100
window_length = n_steps + 1 #target = input shifted 1 chacater ahead
dataset = dataset.window(window_length, shift=1, drop_remainder=True)

dataset = dataset.flat_map(lambda window: window.batch(window_length))

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:,:-1], windows[:,1:]))

dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
    
#Building and Training the Char-RNN Model

model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, input_shape=[None,max_id],
                     dropout=0.2, recurrent_dropout=0.2),
    keras.layers.GRU(128, return_sequences=True,
                     dropout=0.2, recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                    activation="softmax"))
    ])

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam")
history = model.fit(dataset, epochs=20)

# Using the Char-RNN Model

def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)

X_new = preprocess(["How are yo"])
Y_pred = model.predict_classes(X_new)
tokenizer.sequences_to_texts(Y_pred + 1)[0][-1]


# Sentiment Analysis

(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()
X_train[0][:10]

word_index = keras.datasets.imdb.get_word_index()
id_to_word = {id_+3: word for word, id_ in word_index.items()}
for id_, token in enumerate(("<pad>","<sos>","<unk>")):
    id_to_word[id_] = token
    
" ".join([id_to_word[id_] for id_ in X_train[0][:10]])

import tensorflow_datasets as tfds

# datasets, info = tfds.load("imdb_reviews", as_supervised = True, with_info=True)    
