#Construa um algoritmo que gere um datasheet aleatorio com colunas de altura, peso, idade e genero

import numpy as np
import pandas as pd
from random import randrange, uniform
import random

GeneroList = ['Masculino','Feminino']

df = pd.DataFrame()
COLUNAS = [
    'Genero',
    'Altura',
    'Idade',
    'Peso'
]
df = pd.DataFrame(columns=COLUNAS)

for x in range(100):
    genero = random.choice(GeneroList)
    altura = uniform(1.50,2.0)
    idade = randrange(15,60)
    peso = uniform(40.0,100.0)
    
    df.loc[-1] = [genero, altura, idade, peso]
    df.index = df.index + 1 
    df = df.sort_index()

df['Idade'] = df['Idade'].astype(int)

print(df)

df.to_csv('cadastro.csv')
