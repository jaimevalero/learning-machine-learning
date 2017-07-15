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

#X = df.iloc[:, [4,5,6]]
X = df.iloc[:, [
  df.columns.get_loc("acceleration_x"),
  df.columns.get_loc("acceleration_y"),
  df.columns.get_loc("acceleration_z")
]]


df.columns.get_loc("username")
y = df.iloc[:, [3]]

# Separamos los datos de train
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)


# Estandarizamos el train y test (0,1) y desviación tipica
from sklearn.svm import SVC

pipe_svc = Pipeline([('scl', StandardScaler()),
            ('clf', SVC(random_state=1))])
#param_grid=[{'clf__C': 1000.0, 'clf__kernel': 'rbf'}]
pipe_svc.set_params(clf__C=1000, clf__kernel='rbf')
clf = pipe_svc.fit(X_train, y_train.values.ravel())

y_pred = pipe_svc.predict(X_test)



print('Test accuracy: %.3f' % clf.score(X_test, y_test))
print('Misclassified samples: %d' % (y_test.values.ravel() != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# Ejemplo para predecir
df_ = pd.DataFrame([[0.265,0.365,0.065]],columns=['acceleration_x','acceleration_y', 'acceleration_z'])
pipe_svc.predict(df_)

kk = pipe_svc.predict(df_)
kk[0]
