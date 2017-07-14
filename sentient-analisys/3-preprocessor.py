import re
import pandas as pd
import numpy as np

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + \
           ' '.join(emoticons).replace('-', '')
    return text

df = pd.read_csv('./movie_data.csv')

preprocessor(df.loc[0, 'review'][-50:])
print (preprocessor(df.loc[0, 'review'][-50:]))

df['review'] = df['review'].apply(preprocessor)

from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
    return text.split()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


tokenizer('runners like running and thus they run')

tokenizer_porter('runners like running and thus they run')

import nltk
nltk.download('stopwords')


from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]
