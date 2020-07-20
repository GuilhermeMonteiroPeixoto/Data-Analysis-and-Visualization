# # Analisando Breast Cancer Wisconsin com Random Forest

import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
data2 = pd.DataFrame(data['data'], columns = data['feature_names'] )

labels = pd.DataFrame(data['target'], columns = ['Result'])
df_join = data2.join(labels)

y = df_join['Result']
x = df_join.drop('Result',axis=1)

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size = 0.1)

from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier()
modelo.fit(x_treino,y_treino)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

print('Erro medio absoluto: ',mean_absolute_error(y_teste,previsao))
print('Acuracia: ',accuracy_score(y_teste,previsao))



