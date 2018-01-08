import numpy as np
import Perceptron
import random

"""
This program uses the Perceptron to check if a point is above or below a line
x = y 

Input type, different x and y values
"""

data = []   # data to be given as input for fit method
real_output = []    # output for each individual input

for i in range(-500, 501):
    x_value = i
    y_value = i + random.randint(-5, 5)
    data.append([x_value, y_value])
    out = 1 if y_value - x_value >= 0 else -1
    real_output.append(out)

# Converting it to a numpy array
data = np.array(data)
real_output = np.array(real_output)

point_predictor = Perceptron.Perceptron(learning_rate=0.01, iterations=len(real_output))
point_predictor.fit(data, real_output)

a, b = list(map(int, (input("Enter a point in form 1 2 for(1,2):").split())))
result = point_predictor.predict(np.array([a, b]))

if result == 1:
    print("The point ({},{}) is above or on the line y=x".format(a, b))
else:
    print("The point ({},{}) is below the line y=x".format(a, b))