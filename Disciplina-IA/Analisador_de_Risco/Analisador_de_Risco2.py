import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

#Leitura do Arquivo
dataset = pd.read_csv('Risco_Credito4.csv')
print("Leitura da Base de Dados [OK]")

#Limpeza da Base de Dados
dataset = dataset.drop(['Pontos'], axis=1)
#Removendo outliers
dataset = dataset[dataset["Idade"]<=100]
dataset = dataset[dataset["Tempo_Emprego"]<=70]
dataset = dataset[dataset["Tempo_Devolucao"]<= 200]
dataset = dataset[dataset["Valor_Empre"]>= 0]
print("Limpeza da Base de Dados [OK]")

#Dividindo base de dados para Treino
y = dataset['Risco']
x = dataset.drop('Risco',axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size = 0.25)
modelo = RandomForestClassifier()
modelo.fit(x_treino,y_treino)
print("Construção do Modelo com Random Forest [OK]")

#Montando Arvore de Decisao
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

y = dataset['Risco']
x = dataset.drop('Risco',axis=1)

clf = DecisionTreeClassifier(random_state=1234, max_depth=5)
model = clf.fit(x, y)

print("Construção do Modelo com Arvore de Decisao [OK]")

fn=['Idade', 'Residencia', 'Salario', 'Tempo_Emprego', 'Servidor_Pub',
       'Grau_Edu', 'Montante', 'Nome_Limpo', 'Valor_Empre', 'Tempo_Devolucao']
cn=['Alto', 'Moderado', 'Baixo']

fig = plt.figure(figsize=(50,40))
tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')


print("Arvore de Decisao [OK]")

#Testando Arvore de Decisao
previsao2 = clf.predict(x_teste)
print('Arvore de Decisao - Acuracia : ',accuracy_score(y_teste, previsao2))
print('Arvore de Decisao - Precisao : ',precision_score(y_teste, previsao2, average='macro'))

#Testando Modelo
previsao = modelo.predict(x_teste)
print('Random Forest - Acuracia : ',accuracy_score(y_teste, previsao))
print('Random Forest - Precisao : ',precision_score(y_teste, previsao, average='macro'))

print("\n\nInforme os seus dados\n")


#Mostrando features Importantes
importances = modelo.feature_importances_
indices = np.argsort(importances)[::-1]
variable_importance = {'importance': importances,
            'index': indices}
importances_modelo = variable_importance['importance']
indices_modelo = variable_importance['index']

names_index = x_teste.columns

index = np.arange(len(names_index))
importance_desc = sorted(importances)
feature_space = []
for i in range(indices.shape[0] - 1, -1, -1):
    feature_space.append(names_index[indices[i]])
fig, ax = plt.subplots(figsize=(10, 10))
plt.title('Importancia para Random Forest')
plt.barh(index,importance_desc,align="center")
plt.yticks(index,feature_space)
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature')
plt.show()

#Entrada dos dados do User
def analise_risco():
    Idade_Entrada = input("Digite a sua Idade: ")
    Residencia_Entrada = input("1.Residencia Própria 2.Alugada 3.Outros: ")
    Salario_Entrada = input("Salário Inteiro: ")
    Tempo_Emprego_Entrada = input("Quantos meses você está no Trabalho: ")
    Servidor_Pub_Entrada = input("Servidor Público (0 para Não e 1 para Sim):")
    Grau_Edu_Entrada = input("Grau de Educacao (1,2,3,4):")
    Montante_Entrada = input("Quanto você tem na conta: ")
    Nome_Limpo_Entrada = input("Nome Limpo (0 para Não e 1 para Sim):")
    Valor_Empre_Entrada = input("Valor Emprestimo:")
    Tempo_Devolucao_Entrada = input("Meses para Devoluçao:")
    
    array = [[Idade_Entrada],[Residencia_Entrada],[Salario_Entrada],[Tempo_Emprego_Entrada],[Servidor_Pub_Entrada],[Grau_Edu_Entrada],[Montante_Entrada],[Nome_Limpo_Entrada],[Valor_Empre_Entrada],[Tempo_Devolucao_Entrada]]
    data_array = pd.DataFrame(array, index = ['Idade', 'Residencia', 'Salario', 'Tempo_Emprego', 'Servidor_Pub',
           'Grau_Edu', 'Montante', 'Nome_Limpo', 'Valor_Empre', 'Tempo_Devolucao'])
    
    data_array = data_array.T

    previsao2 = clf.predict(data_array)
    previsao = modelo.predict(data_array)
    return previsao2

print("\n")
pontuacao = int(analise_risco())

if pontuacao<1:
    print("O risco é alto.")
elif pontuacao==1:
    print("O risco é moderado.")
else:
    print("O risco é baixo.")

input()
