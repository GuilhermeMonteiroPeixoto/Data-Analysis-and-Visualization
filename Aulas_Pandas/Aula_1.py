#AULA 1 PANDAS

import pandas as pd
import numpy as np

#Criando dataframe
lista = {'nome':['Julia','Anne','Carol','Felipe','Carlos','Guilherme','Thiago'],
         'genero':['f','f','f','m','m','m','m'],
         'Nota 1':[np.nan, 4, 8, 9, 8, 8, 7],
         'Nota 2':[6, 9, np.nan, 7, 7, 7, 7],
         'Nota 3':[5, 7, 7, np.nan, 9, 10, 9]}
data_list = pd.DataFrame(lista, index = ['29384','34211','34172','06874','23243','16341','34713'])
print(data_list,'\n\n')]

#Comandos Basicos
print("Média:", data_list['Nota 1'].mean(),'\n\n')
print("Desvio padrão:", data_list['Nota 1'].std(),'\n\n')
print(data_list.dtypes,'\n\n')
print(data_list.columns,'\n\n')
print(data_list.index,'\n\n')
#Resumo estatístico do DataFrame, com quartis, mediana, etc.
print(data_list.describe(),'\n\n')
print(data_list.sort_values(by='nome'),'\n\n')
print(data_list['genero'].unique(),'\n\n')
print(data_list['genero'].value_counts(),'\n\n')
#Soma dos valores de um DataFrame
print(data_list['Nota 1'].sum(),'\n\n')
#Menor valor de um DataFrame
print(data_list['Nota 1'].min(),'\n\n')
#Maior valor
print(data_list['Nota 1'].max(),'\n\n')
#Média dos valores
print(data_list['Nota 1'].mean(),'\n\n')
#Mediana dos valores
print(data_list['Nota 1'].median(),'\n\n')
