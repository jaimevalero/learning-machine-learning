import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv('wdbc.data',
       header=None)

from sklearn.preprocessing import LabelEncoder
# We assign the 30 features to a NumPy array X.
X = df.loc[:, 2:].values
y = df.loc[:, 1].values



# Using LabelEncoder, we transform the class labels
# from their original string representation
# (M and B) into integers:
le = LabelEncoder()
y = le.fit_transform(y)
'''After encoding the class labels (diagnosis) in an array y,
the malignant tumors are now represented as class 1,
and the benign tumors are represented as class 0, respectively,
which we can illustrate by calling the transform method of LabelEncoder
on two dummy class labels:
'''
le.transform(['M', 'B'])

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)



from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

pipe_svc = Pipeline([('scl', StandardScaler()),
            ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
                 {'clf__C': param_range,
                  'clf__gamma': param_range,
                  'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=2)

gs = gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)



clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))



