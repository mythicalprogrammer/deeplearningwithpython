#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:00:06 2017

@author: anthonydoan
"""

# workign with Reuters dataset, a set of short newswires and their topics,
# published by Reuters in 1986.
# 46 topics, each topic has at least 10 examples in training set

from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

len(train_data) # 8982
len(test_data) # 2246

print(train_data[10])
"""
output
[1, 245, 273, 207, 156, 53, 74, 160, 26, 14, 46, 296, 26, 39, 74, 2979, 3554, 
14, 46, 4689, 4329, 86, 61, 3499, 4795, 14, 61, 451, 4329, 17, 12]
"""

# decode the vector back to words to read it
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# Note that our indices were offset by 3 
# because 0, 1, and 2 are reserved indices for "padding", "start of sequence",
# and "unknow".
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') 
for i in train_data[0]])

# label associated 
print(train_labels[10]) # 3
