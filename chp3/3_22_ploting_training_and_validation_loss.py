# -*- coding: utf-8 -*-
# ploting training and validation loss
import matplotlib.pyplot as plt

"""
blue dots bo are training loss 
blue crosses are validation loss
"""

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss_values, 'bo')
# b+ is for "blue crosses"
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

"""
You can see overfitting as the blue dot gets more
accurate via training data.

Your cross validation gets worst because you are
overfitting, fitting so perfectly to the training data
that your model is terrible at generalizing when
non train data is used to predict. 
"""

