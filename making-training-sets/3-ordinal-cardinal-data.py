
import pandas as pd


from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame([
['green', 'M', 10.1, 'class1'],
['red', 'L', 13.5, 'class2'],
['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']
# DataFrame original
print(df)

size_mapping = {
'XL': 3,
'L': 2,
'M': 1}
df['size'] = df['size'].map(size_mapping)
# DataFrame con la columna size mapeada a valores numéricos
print(df)

# If we want to transform the integer values back to the original string
# representation at a later stage, we can simply de ne a reverse-mapping
# dictionary
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(size_mapping)
print(inv_size_mapping)
df['size'] = df['size'].map(inv_size_mapping)
# Deshacemos los cambios
print(df)


# Class label encoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
df['classlabel'] = y

print(df)
