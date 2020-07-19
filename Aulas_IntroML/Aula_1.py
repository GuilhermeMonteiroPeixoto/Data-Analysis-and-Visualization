# AULA 1 ML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

dados_estudantes = pd.read_csv("Dados_estudante.csv")

y = dados_estudantes['nota_redacao']

x = dados_estudantes.drop('nota_redacao',axis=1)

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size = 0.3)

from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import ExtraTreesRegressor
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor


modelo = ExtraTreesClassifier()
#modelo = ExtraTreesRegressor(n_estimators=100, random_state=0)
#modelo = RandomForestClassifier()
#modelo = RandomForestRegressor()

modelo.fit(x_treino,y_treino)

previsao = modelo.predict(x_teste[0:30])

print('Erro: ',mean_absolute_error(y_teste[0:30],previsao))
print('Acuracia: ',accuracy_score(y_teste[0:30],previsao))

plt.plot(previsao)
plt.plot(np.array(y_teste[0:60]))
plt.show()

