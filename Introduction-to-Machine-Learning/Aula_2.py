# # Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([(2*a+3) for a in range(1000)])
y = np.array([(200**np.random.random() + b)for b in range(1000)])

plt.plot(x,y,'o')
plt.title('X vs Y')  
plt.xlabel('X')  
plt.ylabel('Y')
plt.show()

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

x = x.reshape(-1,1)
y = y.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

regressor = LinearRegression()  
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x, y)
plt.plot(x_test, y_pred, color='red', linewidth=2)
plt.show()
