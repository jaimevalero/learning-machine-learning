




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


pipe_lr = Pipeline([('scl', StandardScaler()),
   ('pca', PCA(n_components=2)),
   ('clf', LogisticRegression(C=500,random_state=0))])
pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))



from sklearn.learning_curve import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
param_range = [ 0.1, 0.2,  0.3, 0.4, 0.5, 0.6, ]

train_scores, test_scores = validation_curve(
                estimator=pipe_lr,
                X=X_train,
                y=y_train,
                param_name='clf__C',
                param_range=param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.tight_layout()
plt.savefig('validation_curve.png', dpi=300)
plt.show()
