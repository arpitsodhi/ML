# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('kc_house_data.csv')
X = dataset.iloc[:, 2:9].values
y = dataset.iloc[:, 2].values

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)

# Predicting a new result
y_pred = regressor.predict(x_test)
print("Root Mean squared error: %.2f" % np.sqrt(np.mean((regressor.predict(x_train) - y_train) ** 2)))
print("Root Mean squared error: %.2f" % np.sqrt(np.mean((regressor.predict(x_test) - y_test) ** 2)))
print('Variance score: %.2f' % regressor.score(x_test, y_test))