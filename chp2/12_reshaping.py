import numpy as np

x = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])
print(x.shape)

x.reshape((6, 1))
x.reshape((1, 6))
x.reshape((2, 3))