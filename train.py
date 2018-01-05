import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import Perceptron


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


df = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data', header=None)  # Loading the iris dataset

y = df.iloc[0:100, 4].values  # gets all rows in 4th column
y = np.where(y == 'Iris-setosa', -1, 1)   # assigns -1 to Iris-setosa and 1 to Iris-versicolor

X = df.iloc[0:100, [0, 2]].values

# Viewing the dataset in Matplotlib
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:, 0], X[50:, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('Sepel length')
plt.ylabel('Petel length')
plt.legend(loc='upper left')

plt.show()

# Training the perceptron
perceptron = Perceptron.Perceptron()
perceptron = perceptron.fit(X, y)

# Shows how the perceptron improves after calling fit in n_iterations
plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('No of misclassifications')
plt.show()

plot_decision_regions(X, y, classifier=perceptron)
plt.xlabel('sepel length[cm]')
plt.ylabel('petel length[cm]')
plt.legend(loc='upper left')
plt.show()
