import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('99_Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
 
from sklearn.model_selection import train_test_split
x_train 