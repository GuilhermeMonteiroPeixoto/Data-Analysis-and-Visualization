import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re

def similarityCalculate(source, text): 
 
  # N-gramas
  n = 1
  # Instancia o contador de n-gramas
  counts = CountVectorizer(analyzer = 'word', ngram_range=(n,n))
  
  # Cria uma matriz de contagem de n-grama para os dois textos
  n_grams = counts.fit_transform([text, source])
 
  # Cria um dicionário de n-gramas
  vocab2int = counts.fit([text, source]).vocabulary_
 
  n_grams_array = n_grams.toarray()
 
  #interceção entre os textos (containment)
  intersection_list = np.amin(n_grams.toarray(), axis = 0)
  #print(intersection_list,'\n')
 
  intersection_count = np.sum(intersection_list)
  #print(intersection_count)
 
  index_A = 0
  A_count = np.sum(n_grams.toarray()[index_A])
  #print(A_count)
 
  normal = intersection_count/A_count
 
  # Printa dicionário de palavras: index
  #print(vocab2int)
  #print('Vetor de n-gramas:\n\n', n_grams_array)
  #print()
  #print('Dicionário de n-gramas (unigrama):\n\n', vocab2int)
  return normal
 
def similarityDegree(source, text):
  similarityType = 0
  degree = 0
  
  for i in source:
    similarity = similarityCalculate(i, text)
    if degree < similarity:
      degree = similarity
      similarityType = i
  return similarityType

def idade_similaridade():
  age = input('Qual é a sua idade? (Ex: 25, 25 anos, eu tenho 25 anos)')
  return re.search(r'\d+', age).group(0)

def residencia_similaridade():
  similarity = input('Em que tipo de residência você mora? (Ex: Própria, Alugada, Outros)')
  similarityType = similarityDegree([
    'Outros',
    'Alugada',
    'Residência Própria'
    ], similarity)
   
  if similarityType == 'Residência Própria': return 1
  elif similarityType == 'Alugada': return 2
  else: return 3

def salario_similaridade():
  salario = input('Qual é a sua renda mensal?')
  return re.search(r'\d+', salario).group(0)

def tempo_emprego_similaridade():
  tempo_emprego = input('Há quanto tempo você trabalha no emprego atual (em meses)?')
  return re.search(r'\d+', tempo_emprego).group(0)

def servidor_pub_similaridade():
  servidor = input('Você é servidor público? (Sim/Não)').lower()
  if servidor == 'sim': return 1
  else: return 0

def grau_edu_similaridade():
  similarity = input('Qual é a sua formação educacional? (Ex: Fundamental, Médio, Superior completo)')
  similarityType = similarityDegree([
    'Ensino Médio',
    'Ensino Fundamental',
    'Ensino Superior Completo',
    'Ensino Superior Incompleto'
    ], similarity)
  
  if similarityType == 'Ensino Fundamental': return 1
  elif similarityType == 'Ensino Médio': return 2
  elif similarityType == 'Ensino Superior Incompleto': return 3
  else: return 4

def montante_similaridade():
  montante = input("Quanto você tem de valor na sua conta?: ")
  return re.search(r'\d+', montante).group(0)  

def nome_limpo_similaridade():
  nome_limpo = input('O seu nome está limpo? (Sim/Não)').lower()
  if nome_limpo == 'sim': return 1
  else: return 0

def valor_empre_similaridade():
  valor_emprestimo = input('Quanto você deseja de empréstimo?')
  return re.search(r'\d+', valor_emprestimo).group(0)
  return valor_emprestimo

def tempo_devolucao_similaridade():
  tempo_devolucao = input('Em quanto tempo deseja devolver o dinheiro? (em meses)')
  return re.search(r'\d+', tempo_devolucao).group(0)

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
    Idade_Entrada = idade_similaridade()
    Residencia_Entrada = residencia_similaridade()
    Salario_Entrada = salario_similaridade()
    Tempo_Emprego_Entrada = tempo_emprego_similaridade()
    Servidor_Pub_Entrada = servidor_pub_similaridade()
    Grau_Edu_Entrada = grau_edu_similaridade()
    Montante_Entrada = montante_similaridade()
    Nome_Limpo_Entrada = nome_limpo_similaridade()
    Valor_Empre_Entrada = valor_empre_similaridade()
    Tempo_Devolucao_Entrada = tempo_devolucao_similaridade()
    
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
