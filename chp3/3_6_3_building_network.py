# -*- coding: utf-8 -*-

from keras import models
from keras import layers

"""
Because so few samples are available, we will be using a very small network 
with two hidden layers, each with 64 units.
"""
def build_model():
    # Because we will need to instantiate
    # the same model multiple time.
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],))) # shape[1] we're getting the number of columns/features
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model