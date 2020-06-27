#AULA 0 PANDAS

import pandas as pd
import numpy as np

#Criando dataframe a partir de array
array = [['Julia','Anne','Carol','Felipe'],['f','f','f','m'],[10., 9., np.nan, 9.5],
         [3, 4, 5, np.nan],[np.nan, 4, 8, 9]]
data_array = pd.DataFrame(array, index = ['nome','genero','nota 1','nota 2','nota 3'],
                          columns = ['29384','34211','34172','06874'])
data_array_T = data_array.T
print(data_array_T,'\n\n')


#Criando dataframe a partir de lista
lista = {'nome':['Julia','Anne','Carol','Felipe'],
         'genero':['f','f','f','m'],
         'Nota 1':[np.nan, 4, 8, 9],
         'Nota 2':[6, 9, np.nan, 7],
         'Nota 3':[5, 7, 7, np.nan]}
data_list = pd.DataFrame(lista, index = ['29384','34211','34172','06874'])
print(data_list,'\n\n')



