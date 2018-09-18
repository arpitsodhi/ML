import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



dataset = pd.read_csv('100_student24.csv')

dataset.insert(0, 'b0', np.ones(1000))

x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,2].values


B = np.array([0,0,0])

n = len(x[:,0])

def cost_function(x,y,B):
    cost = np.sum((x.dot(B) - y) ** 2) / (2 * n)
    return cost

y_pred = x.dot(B)


def gradient_descent(x,y,B,epoch,learning_rate):
    cost_history = [0] * epoch
    
    for i in range(epoch):
        h = x.dot(B)
        loss = h - y
        gradient = x.T.dot(loss) / n
        B = B - learning_rate * gradient
        cost = cost_function(x,y,B)
        cost_history[i] = cost
        
    return cost_history,B


cost, newB = gradient_descent(x,y,B,1000,learning_rate = 0.0001)

y_pred = x.dot(newB)


plt.plot([i for i in range(1000)], cost)
plt.show()