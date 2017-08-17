#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:10:38 2017

@author: anthonydoan
"""

# Cannot feed list of integers into neural network
# needs to convert to tensors

"""
We could one-hot-encode our lists to turn them into vectors of 0s and 1s.
"""

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    # https://docs.python.org/2.3/whatsnew/section-enumerate.html
    # https://stackoverflow.com/questions/22171558/what-does-enumerate-mean
    for i, sequence in enumerate(sequences):
        print("i: {} & sequence: {}".format(i, sequence))
        results[i, sequence] = 1. # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data) 
# 25000, 10000 2D tensor basically there are 25000 rows in train data & 10000 words
# for each row in the train data there is 10000 columns to represent each word
# 0 for if the word is not in that row and 1 if the words are in that row
print(x_train.shape) 
# Our vectorized test data
x_test = vectorize_sequences(test_data)

# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test  = np.asarray(test_labels).astype('float32')

"""
output 
[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 
173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4,
172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 
192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 
16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 
62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 
12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 
77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 
723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 
381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 
476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 
16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 
178, 32]
"""
print(train_data[0])
len(train_data[0]) # 218

vectorize_sequences(train_data[0])
"""
Output
i: 0 & sequence: 1
i: 1 & sequence: 14
i: 2 & sequence: 22
i: 3 & sequence: 16
i: 4 & sequence: 43
i: 5 & sequence: 530
i: 6 & sequence: 973
i: 7 & sequence: 1622
i: 8 & sequence: 1385
i: 9 & sequence: 65
i: 10 & sequence: 458
i: 11 & sequence: 4468
i: 12 & sequence: 66
i: 13 & sequence: 3941
i: 14 & sequence: 4
i: 15 & sequence: 173
i: 16 & sequence: 36
i: 17 & sequence: 256
i: 18 & sequence: 5
i: 19 & sequence: 25
i: 20 & sequence: 100
i: 21 & sequence: 43
i: 22 & sequence: 838
i: 23 & sequence: 112
i: 24 & sequence: 50
i: 25 & sequence: 670
i: 26 & sequence: 2
i: 27 & sequence: 9
i: 28 & sequence: 35
i: 29 & sequence: 480
i: 30 & sequence: 284
i: 31 & sequence: 5
i: 32 & sequence: 150
i: 33 & sequence: 4
i: 34 & sequence: 172
i: 35 & sequence: 112
i: 36 & sequence: 167
i: 37 & sequence: 2
i: 38 & sequence: 336
i: 39 & sequence: 385
i: 40 & sequence: 39
i: 41 & sequence: 4
i: 42 & sequence: 172
i: 43 & sequence: 4536
i: 44 & sequence: 1111
i: 45 & sequence: 17
i: 46 & sequence: 546
i: 47 & sequence: 38
i: 48 & sequence: 13
i: 49 & sequence: 447
i: 50 & sequence: 4
i: 51 & sequence: 192
i: 52 & sequence: 50
i: 53 & sequence: 16
i: 54 & sequence: 6
i: 55 & sequence: 147
i: 56 & sequence: 2025
i: 57 & sequence: 19
i: 58 & sequence: 14
i: 59 & sequence: 22
i: 60 & sequence: 4
i: 61 & sequence: 1920
i: 62 & sequence: 4613
i: 63 & sequence: 469
i: 64 & sequence: 4
i: 65 & sequence: 22
i: 66 & sequence: 71
i: 67 & sequence: 87
i: 68 & sequence: 12
i: 69 & sequence: 16
i: 70 & sequence: 43
i: 71 & sequence: 530
i: 72 & sequence: 38
i: 73 & sequence: 76
i: 74 & sequence: 15
i: 75 & sequence: 13
i: 76 & sequence: 1247
i: 77 & sequence: 4
i: 78 & sequence: 22
i: 79 & sequence: 17
i: 80 & sequence: 515
i: 81 & sequence: 17
i: 82 & sequence: 12
i: 83 & sequence: 16
i: 84 & sequence: 626
i: 85 & sequence: 18
i: 86 & sequence: 2
i: 87 & sequence: 5
i: 88 & sequence: 62
i: 89 & sequence: 386
i: 90 & sequence: 12
i: 91 & sequence: 8
i: 92 & sequence: 316
i: 93 & sequence: 8
i: 94 & sequence: 106
i: 95 & sequence: 5
i: 96 & sequence: 4
i: 97 & sequence: 2223
i: 98 & sequence: 5244
i: 99 & sequence: 16
i: 100 & sequence: 480
i: 101 & sequence: 66
i: 102 & sequence: 3785
i: 103 & sequence: 33
i: 104 & sequence: 4
i: 105 & sequence: 130
i: 106 & sequence: 12
i: 107 & sequence: 16
i: 108 & sequence: 38
i: 109 & sequence: 619
i: 110 & sequence: 5
i: 111 & sequence: 25
i: 112 & sequence: 124
i: 113 & sequence: 51
i: 114 & sequence: 36
i: 115 & sequence: 135
i: 116 & sequence: 48
i: 117 & sequence: 25
i: 118 & sequence: 1415
i: 119 & sequence: 33
i: 120 & sequence: 6
i: 121 & sequence: 22
i: 122 & sequence: 12
i: 123 & sequence: 215
i: 124 & sequence: 28
i: 125 & sequence: 77
i: 126 & sequence: 52
i: 127 & sequence: 5
i: 128 & sequence: 14
i: 129 & sequence: 407
i: 130 & sequence: 16
i: 131 & sequence: 82
i: 132 & sequence: 2
i: 133 & sequence: 8
i: 134 & sequence: 4
i: 135 & sequence: 107
i: 136 & sequence: 117
i: 137 & sequence: 5952
i: 138 & sequence: 15
i: 139 & sequence: 256
i: 140 & sequence: 4
i: 141 & sequence: 2
i: 142 & sequence: 7
i: 143 & sequence: 3766
i: 144 & sequence: 5
i: 145 & sequence: 723
i: 146 & sequence: 36
i: 147 & sequence: 71
i: 148 & sequence: 43
i: 149 & sequence: 530
i: 150 & sequence: 476
i: 151 & sequence: 26
i: 152 & sequence: 400
i: 153 & sequence: 317
i: 154 & sequence: 46
i: 155 & sequence: 7
i: 156 & sequence: 4
i: 157 & sequence: 2
i: 158 & sequence: 1029
i: 159 & sequence: 13
i: 160 & sequence: 104
i: 161 & sequence: 88
i: 162 & sequence: 4
i: 163 & sequence: 381
i: 164 & sequence: 15
i: 165 & sequence: 297
i: 166 & sequence: 98
i: 167 & sequence: 32
i: 168 & sequence: 2071
i: 169 & sequence: 56
i: 170 & sequence: 26
i: 171 & sequence: 141
i: 172 & sequence: 6
i: 173 & sequence: 194
i: 174 & sequence: 7486
i: 175 & sequence: 18
i: 176 & sequence: 4
i: 177 & sequence: 226
i: 178 & sequence: 22
i: 179 & sequence: 21
i: 180 & sequence: 134
i: 181 & sequence: 476
i: 182 & sequence: 26
i: 183 & sequence: 480
i: 184 & sequence: 5
i: 185 & sequence: 144
i: 186 & sequence: 30
i: 187 & sequence: 5535
i: 188 & sequence: 18
i: 189 & sequence: 51
i: 190 & sequence: 36
i: 191 & sequence: 28
i: 192 & sequence: 224
i: 193 & sequence: 92
i: 194 & sequence: 25
i: 195 & sequence: 104
i: 196 & sequence: 4
i: 197 & sequence: 226
i: 198 & sequence: 65
i: 199 & sequence: 16
i: 200 & sequence: 38
i: 201 & sequence: 1334
i: 202 & sequence: 88
i: 203 & sequence: 12
i: 204 & sequence: 16
i: 205 & sequence: 283
i: 206 & sequence: 5
i: 207 & sequence: 16
i: 208 & sequence: 4472
i: 209 & sequence: 113
i: 210 & sequence: 103
i: 211 & sequence: 32
i: 212 & sequence: 15
i: 213 & sequence: 16
i: 214 & sequence: 5345
i: 215 & sequence: 19
i: 216 & sequence: 178
i: 217 & sequence: 32
"""