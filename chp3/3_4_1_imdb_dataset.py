#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:57:14 2017

@author: anthonydoan
"""

from keras.datasets import imdb
# The argument num_words=10000 means that we will only keep the top 10,000 
# most frequently occurring words in the training data. 
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data[0])
print(train_labels[0])

"""
Since we restricted ourselves to the top 10,000 most frequent words, 
no word index will exceed 10,000 
"""
max_num_words = max([max(sequence for sequence in train_data)])
print(max_num_words)

"""
Decoding the integer sequences back into sentences

"""
# word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# We reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# We decode the review; note that our indices were offset by 3
# because 0, 1, and 2 are reserved incides for "padding", "start of sequences",
# and "unknown".
decode_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

