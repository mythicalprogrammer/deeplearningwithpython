import numpy as np

# x is a random tensor with shape (64, 3, 32, 10)
x = np.random.random((64, 3, 32, 10))
# y is a random tensor with shape (32, 10)
y = np.random.random((32, 10))

# The output z has shape (64, 3, 32, 10) like x
z = np.maximum(x, y)
z.shape
