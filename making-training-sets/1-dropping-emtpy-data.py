>>> import pandas as pd
>>> from io import StringIO
>>> 
>>> csv_data = '''A,B,C,D
... 1.0,2.0,3.0,4.0
... 5.0,6.0,,8.0
... 10.0,11.0,12.0,'''
>>> 
>>> df = pd.read_csv(StringIO(csv_data))
>>> print(df)
      A     B     C    D
0   1.0   2.0   3.0  4.0
1   5.0   6.0   NaN  8.0
2  10.0  11.0  12.0  NaN
>>> df.isnull().sum()
A    0
B    0
C    1
D    1
dtype: int64

>>> df.isnull()
       A      B      C      D
0  False  False  False  False
1  False  False   True  False
2  False  False  False   True

>>> df.dropna()
     A    B    C    D
0  1.0  2.0  3.0  4.0

>>> df.dropna(axis=1)
      A     B
0   1.0   2.0
1   5.0   6.0
2  10.0  11.0

>>> df.dropna(axis=0)
     A    B    C    D
0  1.0  2.0  3.0  4.0

>>> # only drop rows where all columns are NaN
... df.dropna(how='all')
      A     B     C    D
0   1.0   2.0   3.0  4.0
1   5.0   6.0   NaN  8.0
2  10.0  11.0  12.0  NaN

>>> #drop rows that have not at least 4 non-NaN values
... df.dropna(thresh=4)
     A    B    C    D
0  1.0  2.0  3.0  4.0

>>> # only drop rows where NaN appear in specific columns (here: 'C')
... df.dropna(subset=['C'])
      A     B     C    D
0   1.0   2.0   3.0  4.0
2  10.0  11.0  12.0  NaN
>>> 

