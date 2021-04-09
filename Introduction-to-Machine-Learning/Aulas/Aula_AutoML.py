#!/usr/bin/env python
# coding: utf-8

# # Identificando os melhores Hiperparâmetros para RFC
# 
# Se você está iniciando seus estudos em algoritmos de regressão e classificação, é provavel que já tenha se perguntado qual a melhor configuração de hipermarâmetros do algoritmo escolhido para obter o melhor modelo possível.
# 
#     R: Depende do seu DataFrame. Mas você sempre pode testar vários parâmetros através de busca exaustiva.
# 

# ## Ajuste de hiperparâmetros através de Busca Exaustiva usando Random Search

# ### Importar Bibliotecas

# In[2]:


import pandas as pd #leitura e manipulação de dataframes
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler #normalização
import time
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.svm import SVC


# ### Importar DataSet

# In[3]:


base = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', sep=',', header=None)


# In[4]:


#colunas da base
preditores = ['id','diagnostic','radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
 'smoothness_mean', 'compactness_mean', 'concavity_mean',
 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
 'smoothness_se', 'compactness_se', 'concavity_se',
 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
 'smoothness_worst', 'compactness_worst', 'concavity_worst',
 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
base.columns = preditores


# ### Definir a Classe e as variáveis dependentes
# 
# Separar em "Dados" e "Resultados" -> (x,y)

# In[5]:


#y é a classe e X as variáveis dependentes
X = base.drop(['id','diagnostic'], axis=1).values
classe = base['diagnostic'].values
labelencoder_classe = LabelEncoder() #converte para zeros e uns
y = labelencoder_classe.fit_transform(classe)


# ### Normalizar Dados

# In[6]:


#normalização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)
pd.DataFrame(X).describe()


# ### Dividir em Dados para Treino e Teste

# In[7]:


#divisão entre treino (80%) e teste (20%)
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
 test_size = 0.2,
 random_state = 42)


# ### Definir os hiperparâmetros a ser utilizado

# In[8]:


#conjunto de hiperparâmetros a ser utilizado
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 20, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(1, 15, num = 5)]
max_depth.append(None)
min_samples_split = [2, 3, 4, 5]
min_samples_leaf = [1, 2, 3, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# ### Definir o RandomizedSearchCV

# In[9]:


from sklearn.model_selection import RandomizedSearchCV
rfc = RandomForestClassifier()
#usando um método de cross-validation com 5 partes. Número de iteraçãos =100
RandomizedSearchCV(cv=20, estimator=rfc, param_distributions=random_grid, n_iter=100, verbose=2, n_jobs=-1)


# In[10]:


parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

RandomizedSearchCV(SVC(), parameters, cv=5)


# In[11]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform

parameters2 = {'C': [0.01, 0.1, 1, 10] }

GridSearchCV(LogisticRegression(), parameters2, cv=5)


# ### Executar o RandomizedSearchCV para SVC

# In[17]:


SVC_RS = RandomizedSearchCV(SVC(), parameters, random_state=140).fit(X,y)


# In[18]:


#exibindo e aplicando os melhores parâmetros
params_SVCRS = SVC_RS.best_params_
SVC1 = SVC(**params_SVCRS)


# In[19]:


SVC1


# In[20]:


SVC1.fit(X_treinamento,y_treinamento)


# In[21]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

previsao = SVC1.predict(X_teste)

print('Acuracia : ',accuracy_score(y_teste, previsao))
print('Precisao : ',precision_score(y_teste, previsao, average='macro'))


# ### Executar o RandomizedSearchCV para Logistic Regression

# In[22]:


LR_RS = GridSearchCV(LogisticRegression(), parameters2).fit(X,y)


# In[23]:


#exibindo e aplicando os melhores parâmetros
params_LRRS = LR_RS.best_params_
LR1 = LogisticRegression(**params_LRRS)


# In[24]:


LR1


# In[25]:


LR1.fit(X_treinamento,y_treinamento)


# In[26]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

previsao = LR1.predict(X_teste)

print('Acuracia : ',accuracy_score(y_teste, previsao))
print('Precisao : ',precision_score(y_teste, previsao, average='macro'))


# ### Executar o RandomizedSearchCV para RandomForestClassifier

# In[27]:


clf_RS = RandomizedSearchCV(rfc, random_grid, random_state=130).fit(X,y)


# In[28]:


#exibindo e aplicando os melhores parâmetros
params_RS = clf_RS.best_params_
clf1 = RandomForestClassifier(**params_RS)


# In[29]:


clf1


# In[30]:


clf1.fit(X_treinamento,y_treinamento)


# In[31]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

previsao = clf1.predict(X_teste)

print('Acuracia : ',accuracy_score(y_teste, previsao))
print('Precisao : ',precision_score(y_teste, previsao, average='macro'))


# ## Ajuste de hiperparâmetros através de Busca Exaustiva usando Força Bruta vale a pena?

# In[35]:


from sklearn import svm
from sklearn.model_selection import cross_val_score

n_estimators = [int(x) for x in np.linspace(start = 1, stop = 20, num = 4)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(1, 15, num = 3)]
max_depth.append(None)
min_samples_split = [2, 3]
min_samples_leaf = [1, 2]
bootstrap = [True, False]

aux = 0.
for mx_D in max_depth:
    for n_E in n_estimators:
        for mx_F in max_features:
            for bts in bootstrap:
                for mss in min_samples_split:
                    for msl in min_samples_leaf:
    
                        clf3 = RandomForestClassifier(bootstrap=bts, max_depth=mx_D, min_samples_leaf=msl, min_samples_split=mss, max_features=mx_F, n_estimators=n_E, )

                        clf3.fit(X_treinamento,y_treinamento)

                        from sklearn.metrics import accuracy_score
                        from sklearn.metrics import precision_score

                        previsao = clf3.predict(X_teste)
                        acuraciaV = cross_val_score(clf3, X, y, cv=5, scoring='accuracy')
                        acuracia = acuraciaV.mean()
                        print('max_depth: ', mx_D)
                        print('n_estimators: ', n_E)
                        print('Acuracia M : ',acuracia)
                        print('\n')

                        if acuracia > aux:
                            aux = acuracia
                            clf_aux = clf3

print('Max Acuracia: ', aux)
clf_aux


# In[32]:


get_ipython().system('pip install seaborn')


# In[36]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

previsao = clf_aux.predict(X_teste)

conf_mat = confusion_matrix(y_teste, previsao)
sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ### Fazendo Cross Validation

# In[37]:


from sklearn import svm
from sklearn.model_selection import cross_val_score


# In[38]:


scores1 = cross_val_score(clf_aux, X, y, cv=20, scoring='accuracy')


# In[39]:


scores1.mean()


# In[40]:


scores2 = cross_val_score(clf1, X, y, cv=20, scoring='accuracy')
scores2.mean()


# ### Validação por Recall, Área sob a curva ROC e F1-Score

# In[41]:


from sklearn.metrics import recall_score #recall

recall = recall_score(y_teste, previsao)
print('Recall: ',recall)


# In[42]:


from sklearn.metrics import f1_score #f1-score

f1 = f1_score(y_teste, previsao)
print('F1-Score: %f' % f1)


# In[43]:


from sklearn.metrics import classification_report # metricas de validação

print(classification_report(y_teste, previsao))


# In[44]:


from sklearn.metrics import roc_curve #curva roc
from sklearn.metrics import roc_auc_score #area sob curva roc

# calculate AUC
auc = roc_auc_score(y_teste, previsao)

# estimando as probabilidades
clf_prob = clf1.predict_proba(X_teste)
probs = clf_prob[:, 1]
rfp, rvp, lim = roc_curve(y_teste, probs)

# gráfico da curva roc
plt.plot(rfp, rvp, marker='.', label='KNN',color="orange")
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

# axis labels
plt.xlabel('1- Especificidade')
plt.ylabel('Sensibilidade')
# show the legend
plt.legend()
# show the plot
plt.show()
print("AUC-ROC :",auc)


# ### Implementação de Ajuste de Hiperparâmetros em Regressão Logística, Árvores de Decisão e XGBoost.

# In[ ]:




