#AULA 5 PANDAS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# link studentdata: https://www.kaggle.com/spscientist/students-performance-in-exams
studentdata = pd.read_csv('arquivo.csv')

#Para selecionar entradas NaN, você pode usar pd.isnull()
print('\n\n',studentdata[pd.isnull(studentdata['Nota_1'])])

#Podemos simplesmente substituir cada NaN por outro valor
studentdata.fillna(0, inplace=True)
print('\n\n',studentdata)

#Podemos ter um valor não nulo que gostaríamos de substituir
studentdata['Nome'].replace("Lia", "Maria", inplace=True)
print('\n\n',studentdata)

#Pdemos usar replace para alterar NaN para 0
#replace(np.nan,0, inplace=True)

#rename (), que permite alterar nomes de índices e / ou nomes de colunas
studentdata.rename(columns={'Media':'Nota_Final'}, inplace=True)
print('\n\n',studentdata)

#Combining DataFrame
studentdata2 = pd.read_csv('Arquivo2.csv')
studentdata = pd.read_csv('arquivo.csv')

studentdata_concat = pd.concat([studentdata,studentdata2],ignore_index=True)
print(studentdata_concat)

df_join = studentdata.join(studentdata2, lsuffix='_1ano', rsuffix='_2ano')
print(df_join)
