from keras.datasets import mnist

# Loading the MNIST dataset in Keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# The training data
train_images.shape
len(train_labels)
train_labels

# The test data
test_images.shape
len(test_labels)
test_labels

"""
Workflow as follow:

1st - present our NN with the training data, train_images and train_labels.
2nd - The network will then learn to associate iamges and labels.
3rd - Ask the network to produce predictions for test_images,
      and we will verify if these predictions match the labels from 
      test_labels.
"""

from keras import models
from keras import layers

network = models.Sequential()
# “our network consists of a sequence of two Dense layers” - densely connected
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
# Second layer is a 10-way softmax layer
# it will return an array of 10 probability scores (summing to 1)
# Each score will be the probability that the current digit image
# belongs to one of our 10 digit classes (0,...,9).
network.add(layers.Dense(10, activation='softmax'))

