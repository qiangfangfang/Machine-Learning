import numpy as np
import operator
import pandas as pd

path = 'data/iris.csv'
data = pd.read_csv(path, header=None)
print(data)

X = data[:,:-1]
y = data[:,-1]

from sklearn.model_selection import  train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

