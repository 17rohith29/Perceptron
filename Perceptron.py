'''
    My new year code!!!!!!
    This is a perceptron which predicts whether the given point is above or below the line x-y=0
    INPUT=x y         where x and y are co-ordinate points (x,y)
    comments and suggestions are welcome
    HAPPY NEWYEAR!!!!!!!!!!
'''

import numpy as np
import random


class Perceptron:

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for inp, val in zip(X, y):
                update = self.eta * (val - self.predict(inp))
                self.w_[1:] += update * inp
                self.w_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self

    def predict(self, X):
        output = X.dot(self.w_[1:]) + self.w_[0]
        return np.where(output >= 0, 1, -1)


def main():
    x = []
    y = []
    for i in range(-100, 100):
        a = [i, random.randint(-100, 100)]
        x.append(a)
        output = 0
        if(a[1] >= a[0]):
            output = 1
        else:
            output = -1
        y.append(output)
    x = np.array(x)
    y = np.array(y)

    a = Perceptron(0.1, len(y))
    a = a.fit(x, y)

    z = input('Enter a point in coordinate plane(x,y)-in format x y:').split()
    z = np.array(z, float)

    if(a.predict(z) == 1):
        print('The point {} is above or on the line x=y'.format(z))
    else:
        print('The point {} is below the line x=y'.format(z))


if __name__ == '__main__':
    main()
