import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

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
                    cv=2,
                    n_jobs=2)

scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# comparamos contra un decission tree
from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV( estimator=DecisionTreeClassifier(random_state=0),
             param_grid=[
                  {'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
             scoring='accuracy',
             cv=5)

scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=2)

print('CV accuracy: %.3f +/- %.3f' % (
             np.mean(scores), np.std(scores)))





# Confussion matrix
#from IPython.display import Image
#Image(filename='.06_08.png', width=300)

from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
