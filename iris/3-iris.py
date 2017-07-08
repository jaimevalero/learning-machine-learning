# Perceptron implementation usando el iris dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Perceptron
import plot_decision_regions

df = pd.read_csv('iris.csv', header=None)

# Y son nuestras clases (-1 Iris setosa:1 Iris-versicolor)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# X son nuestras dos caracteristicas (x[0] sepal length, x[1] petal length)
X = df.iloc[0:100, [0, 2]].values

'''
plt.scatter(X[:50, 0], X[:50, 1],  color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()
'''

# Creamos nuestro perceptron
ppn = Perceptron.Perceptron(eta=0.0005, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()



plot_decision_regions.plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
