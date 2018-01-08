import numpy as np
import pandas as pd
import Perceptron
"""
This program takes in data from the famous iris dataset
and finds out if a particular measurement is iris-setosa or Iris-versicolor
"""

# Getting the data
df = pd.read_csv('iris.csv', header=None)

# Making the data usable by converting it to a numpy array
real_outputs = df.iloc[0:100, 4]
df = df.iloc[:100, [0, 1, 2, 3]].values
real_outputs = np.where(real_outputs == 'Iris-setosa', 1, -1)

# Initializing the Perceptron
perceptron = Perceptron.Perceptron(0.05,len(real_outputs))
perceptron.fit(df, real_outputs)

# Prediction
val = list(map(float,input('Enter 4 nos like 1 1 1 1 with the measurements of flower:').split()))
val = np.array(val)
preditct = perceptron.predict(val)

# Output
if preditct == -1:
    print("It is Iris-setosa")
else:
    print("It is Iris-versicolor")