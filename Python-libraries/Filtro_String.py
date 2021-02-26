import numpy as np
import pandas as pd

#How to calculate the number of characters in each word in a series
serie = pd.Series(['guilherme','york','programador','new','monteiro',
                   'peixoto','oliveira','test','trabalho','city','ciencia',
                   'truck','informação','plan','python','car'])

tamanho_palavras = serie.map(lambda x: len(x))

df = pd.concat([serie, tamanho_palavras], axis=1)
df.columns = ['palavras', 'tamanho']
print(df)

#How to convert the first character of each element in a series to uppercase
print('\n\n',df['palavras'].map(lambda x: x.title()))

#How to filter words that contain atleast 2 vowels from a series
from collections import Counter

mask = df['palavras'].map(lambda x: sum([Counter(x.lower()).get(i, 0) for i in list('aeiou')]) >= 2)
print('\n\n',df['palavras'][mask])
