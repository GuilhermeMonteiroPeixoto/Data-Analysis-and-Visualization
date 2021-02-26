# AULA 2 PANDAS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# link studentdata: https://www.kaggle.com/spscientist/students-performance-in-exams
studentdata = pd.read_csv('datasets_74977_169835_StudentsPerformance.csv')

print('\n\nSelecionando a linha 0\n\n')
selecionador_linha = studentdata.iloc[0,:]
print(selecionador_linha)
print('\n\nSelecionando as primeiras 5 linhas\n\n')
selecionador_linha = studentdata.iloc[:5,:]
print(selecionador_linha)
print('\n\nSelecionando as ultimas 5 linhas\n\n')
selecionador_linha = studentdata.iloc[-5:,:]
print(selecionador_linha)
print('\n\nSelecionando intervalo de linhas\n\n')
selecionador_linha = studentdata.iloc[90:95,:]
print(selecionador_linha)
print('\n\nSelecionando varias linhas especificas\n\n')
selecionador_linha = studentdata.iloc[[5,10,15,20,25],:]
print(selecionador_linha)

print('\n\nSelecionando a coluna 0\n\n')
selecionador_coluna = studentdata.iloc[:,0]
print(selecionador_coluna)
print('\n\nSelecionando as primeiras 3 colunas\n\n')
selecionador_coluna = studentdata.iloc[:,3]
print(selecionador_coluna)
print('\n\nSelecionando as ultimas 3 colunas\n\n')
selecionador_coluna = studentdata.iloc[:,-3:]
print(selecionador_coluna)


print('\n\nSelecionando value de uma linha e coluna no dataframe\n\n')
linha_coluna = studentdata.iloc[1,2]
print(linha_coluna)
print('\n\nSelecionando as 5 primeiras linhas das 2 ultimas colunas\n\n')
linha_coluna = studentdata.iloc[:5,-2:]
print(linha_coluna)
print('\n\nSelecionando linhas e colunas especificas\n\n')
linha_coluna = studentdata.iloc[[5,10,20,30],[1,3,5]]
print(linha_coluna)


print('\n\nSelecionando colunas especificas usando o nome delas\n\n')
usando_iloc = studentdata.loc[:5, ['race/ethnicity','reading score', 'writing score']]
print(usando_iloc)
print('\n\nSelecionando linhas e colunas usando o nome delas\n\n')
usando_iloc = studentdata.loc[[5,10,20,25], ['race/ethnicity','reading score', 'writing score']]
print(usando_iloc)

