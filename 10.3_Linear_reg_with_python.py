import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('101_headbrain.csv')

dataset.head()

x =dataset.iloc[:,2].values
y = dataset.iloc[:,3].values


# Using the formula to calculate b1 and b2
numer = 0
denom = 0

for i in range(len(x)):
    numer += (x[i] - np.mean(x)) * (y[i] - np.mean(y))
    denom += (x[i] - np.mean(x)) ** 2

slope = numer / denom
intercept = np.mean(y) - (slope * np.mean(x))

plt.plot(x,y,'o')


plt.plot(x,y,'o')
plt.plot(x, slope*x + intercept,color='r')

y_pred = slope * x + intercept


