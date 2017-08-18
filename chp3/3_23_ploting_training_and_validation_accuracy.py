# -*- coding: utf-8 -*-
# ploting training and validation accuracy

"""
blue dots bo are training accuracy 
blue crosses are validation accuracy
"""

plt.clf() # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()