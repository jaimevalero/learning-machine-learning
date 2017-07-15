from sklearn.linear_model import LogisticRegression

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

# cargamos los datos
# Data From https://www.kaggle.com/vmalyi/run-or-walk

df = pd.read_csv('run_or_walk.csv')
lb=LabelEncoder()
#In [2]: s = pd.Series(['single', 'touching', 'nuclei', 'dusts',
#                       'touching', 'single', 'nuclei'])
#In [3]: s_enc = pd.factorize(s)
lb=LabelEncoder()

X = df.iloc[:, [4,5,6]]
y = df.iloc[:, [3]]

# Separamos los datos de train
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)


# Estandarizamos el train y test (0,1) y desviación tipica
from sklearn.svm import SVC

pipe_svc = Pipeline([('scl', StandardScaler()),
            ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_range = [ 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
               {'clf__C': param_range,
                'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=2)

# {'clf__C': 1000.0, 'clf__kernel': 'rbf'}

gs = gs.fit(X_train, y_train.values.ravel())

print(gs.best_score_)
print(gs.best_params_)



clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
