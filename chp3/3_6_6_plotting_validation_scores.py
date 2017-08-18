# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

plt.plot(range(len(average_mae_history)), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# omit the first 10 data points for scaling reason
plt.plot(range(len(average_mae_history) - 10), average_mae_history[10:])
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

"""
“According to this plot, it seems that validation MAE stops improving 
significantly after after 150 epochs. Past that point, we start overfitting”

Excerpt From: Francois Chollet. “Deep Learning with Python MEAP V05.” iBooks. 
"""