


import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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
   ('clf', LogisticRegression(C=500,random_state=1))])
pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
