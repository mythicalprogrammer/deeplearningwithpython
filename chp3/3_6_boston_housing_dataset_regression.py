#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:01:41 2017

@author: anthonydoan
"""

from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape) # (404, 13)
print(test_data.shape) # (102, 13)

"""
13 features:
    
1  Per capita crime rate.
2  Proportion of residential land zoned for lots over 25,000 square feet.
3  Proportion of non-retail business acres per town.
4  Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
5  Nitric oxides concentration (parts per 10 million).
6  Average number of rooms per dwelling.
7  Proportion of owner-occupied units built prior to 1940.
8  Weighted distances to five Boston employment centres.
9  Index of accessibility to radial highways.
10 Full-value property-tax rate per $10,000.
11 Pupil-teacher ratio by town.
12 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
13 % lower status of the population.

"""

"""
The targets are the median values of owner-occupied homes, in thousands of 
dollars
"""

print(train_targets)