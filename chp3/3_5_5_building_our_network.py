# -*- coding: utf-8 -*-

from keras import models
from keras import layers
"""
If one layer drops some information relevant to the classification problem, 
this information can never be recovered by later layers: each layer can 
potentially become an "information bottleneck".
"""

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# compiling our model

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])