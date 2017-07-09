
import pandas as pd



from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame([
['green', 'M', 10.1, 'Jersey'],
['red', 'L', 13.5, 'Jersey'],
['blue', 'XL', 15.3, 'Pantalones']])
df.columns = ['color', 'size', 'price', 'Prenda']
X = df[['color', 'price']].values


print(df)

df = pd.get_dummies(df)

print(df)
