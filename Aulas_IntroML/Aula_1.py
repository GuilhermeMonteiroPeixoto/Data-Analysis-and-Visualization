# AULA 1 ML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dados_estudantes = pd.read_csv("Dados_estudante.csv")

y = dados_estudantes['nota_redacao']

x = dados_estudantes.drop('nota_redacao',axis=1)

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size = 0.2)

from sklearn.ensemble import ExtraTreesClassifier

modelo = ExtraTreesClassifier()
modelo.fit(x_treino,y_treino)

previsao = modelo.predict(x_teste[0:20])
plt.plot(previsao)
plt.plot(np.array(y_teste[0:20]))
plt.show()


