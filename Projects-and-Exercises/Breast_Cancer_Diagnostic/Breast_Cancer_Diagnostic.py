#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().system('pip install seaborn')
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix


# In[27]:


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', sep=',', header=None)


# In[28]:


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
data.columns = preditores

data


# In[29]:


data['diagnostic'].replace(['B','M'],[0,1],inplace=True)
data.set_index(['id'], inplace = True)
data


# In[30]:


Aux = data['diagnostic'].value_counts()
print('Malignant :', Aux[1])
print('Benign    :', Aux[0])


# In[31]:


heat_map = sns.heatmap(data.corr(),vmax=1, vmin=-1, center=0,
            cmap=sns.diverging_palette(10, 220, as_cmap=True))
plt.show()


# In[32]:


plt.subplot(2,2,1)
m = plt.hist(data[data["diagnostic"] == 1].radius_mean,bins=20,fc = (1,0,1,0.5),label = "Malignant")
b = plt.hist(data[data["diagnostic"] == 0].radius_mean,bins=20,fc = (1,1,0,0.5),label = "Bening")
plt.legend()
plt.xlabel("Radius Mean")
plt.ylabel("Frequency")
plt.title("Histogram of Radius Mean")

plt.subplot(2,2,2)
m = plt.hist(data[data["diagnostic"] == 1].texture_mean,bins=20,fc = (1,0,1,0.5),label = "Malignant")
b = plt.hist(data[data["diagnostic"] == 0].texture_mean,bins=20,fc = (1,1,0,0.5),label = "Bening")
plt.legend()
plt.xlabel("Texture Mean")
plt.ylabel("Frequency")
plt.title("Histogram of Texture Mean")

plt.subplot(2,2,3)
m = plt.hist(data[data["diagnostic"] == 1].compactness_mean,bins=20,fc = (1,0,1,0.5),label = "Malignant")
b = plt.hist(data[data["diagnostic"] == 0].compactness_mean,bins=20,fc = (1,1,0,0.5),label = "Bening")
plt.legend()
plt.xlabel("Compactness Mean")
plt.ylabel("Frequency")
plt.title("Histogram of Compactness Mean")

plt.subplot(2,2,4)
m = plt.hist(data[data["diagnostic"] == 1].concavity_mean,bins=20,fc = (1,0,1,0.5),label = "Malignant")
b = plt.hist(data[data["diagnostic"] == 0].concavity_mean,bins=20,fc = (1,1,0,0.5),label = "Bening")
plt.legend()
plt.xlabel("Concavity Mean")
plt.ylabel("Frequency")
plt.title("Histogram of Concavity Mean")
plt.tight_layout()
plt.show()


# In[33]:


from sklearn.preprocessing import LabelEncoder,StandardScaler #normalização

#y é a classe e X as variáveis dependentes
X = data.drop(['diagnostic'], axis=1).values
classe = data['diagnostic'].values
labelencoder_classe = LabelEncoder() #converte para zeros e uns
y = labelencoder_classe.fit_transform(classe)


# In[34]:


#normalização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)
pd.DataFrame(X).describe()


# In[35]:


#divisão entre treino (80%) e teste (20%)
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[36]:


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


# In[37]:


random_grid2 = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}


# In[38]:


from sklearn.model_selection import RandomizedSearchCV
#usando um método de cross-validation com 5 partes. Número de iteraçãos =100
RandomizedSearchCV(cv=20, estimator=RandomForestClassifier(), param_distributions=random_grid, n_iter=100, verbose=2, n_jobs=-1)


# In[41]:


from sklearn.svm import SVC

RandomizedSearchCV(SVC(), random_grid2, cv=5)


# In[46]:


clf_RS = RandomizedSearchCV(RandomForestClassifier(), random_grid, random_state=130).fit(X,y)


# In[47]:


#exibindo e aplicando os melhores parâmetros
params_RS = clf_RS.best_params_
clf1 = RandomForestClassifier(**params_RS)


# In[48]:


clf1.fit(X_treinamento,y_treinamento)


# In[49]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

previsao = clf1.predict(X_teste)

print('Acuracia : ',accuracy_score(y_teste, previsao))
print('Precisao : ',precision_score(y_teste, previsao, average='macro'))


# In[51]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

previsao = clf1.predict(X_teste)

conf_mat = confusion_matrix(y_teste, previsao)
sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[52]:


from sklearn.metrics import recall_score #recall

recall = recall_score(y_teste, previsao)
print('Recall: ',recall)


# In[53]:


from sklearn.metrics import f1_score #f1-score

f1 = f1_score(y_teste, previsao)
print('F1-Score: %f' % f1)


# In[54]:


from sklearn.metrics import classification_report # metricas de validação

print(classification_report(y_teste, previsao))


# In[56]:


from sklearn.metrics import roc_curve #curva roc
from sklearn.metrics import roc_auc_score #area sob curva roc

# calculate AUC
auc = roc_auc_score(y_teste, previsao)

# estimando as probabilidades
clf_prob = clf1.predict_proba(X_teste)
probs = clf_prob[:, 1]
rfp, rvp, lim = roc_curve(y_teste, probs)

# gráfico da curva roc
plt.plot(rfp, rvp, marker='.', label='RFC',color="orange")
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

# axis labels
plt.xlabel('1- Especificidade')
plt.ylabel('Sensibilidade')
# show the legend
plt.legend()
# show the plot
plt.show()
print("AUC-ROC :",auc)


# In[69]:


SVC_RS = RandomizedSearchCV(SVC(), random_grid2, random_state=140).fit(X,y)


# In[70]:


#exibindo e aplicando os melhores parâmetros
params_SVCRS = SVC_RS.best_params_
SVC1 = SVC(**params_SVCRS)


# In[71]:


SVC1.fit(X_treinamento,y_treinamento)


# In[72]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

previsao2 = SVC1.predict(X_teste)

print('Acuracia : ',accuracy_score(y_teste, previsao2))
print('Precisao : ',precision_score(y_teste, previsao2, average='macro'))


# In[73]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

previsao2 = SVC1.predict(X_teste)

conf_mat = confusion_matrix(y_teste, previsao2)
sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[74]:


from sklearn.metrics import recall_score #recall

recall = recall_score(y_teste, previsao2)
print('Recall: ',recall)


# In[75]:


from sklearn.metrics import f1_score #f1-score

f1 = f1_score(y_teste, previsao2)
print('F1-Score: %f' % f1)


# In[76]:


from sklearn.metrics import classification_report # metricas de validação

print(classification_report(y_teste, previsao2))


# In[84]:


importances = clf1.feature_importances_
indices = np.argsort(importances)[::-1]
variable_importance = {'importance': importances, 'index': indices}
importances_modelo = variable_importance['importance']
indices_modelo = variable_importance['index']


# In[85]:


names_index = x_teste.columns


# In[86]:


index = np.arange(len(names_index))
importance_desc = sorted(importances)
feature_space = []
for i in range(indices.shape[0] - 1, -1, -1):
    feature_space.append(names_index[indices[i]])
fig, ax = plt.subplots(figsize=(10, 10))
plt.title('Feature importances for Random Forest Model')
plt.barh(index,importance_desc,align="center")
plt.yticks(index,feature_space)
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature')
plt.show()


# In[87]:


conf_mat = confusion_matrix(y_teste, previsao)
sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[90]:


modelo = ExtraTreesClassifier(n_estimators=200,max_depth=5)
modelo.fit(x_treino,y_treino)


# In[92]:


previsao = modelo.predict(X_teste)
print('Acuracia : ',accuracy_score(y_teste, previsao))
print('Precisao : ',precision_score(y_teste, previsao, average='macro'))


# In[93]:


importances = modelo.feature_importances_
indices = np.argsort(importances)[::-1]
variable_importance = {'importance': importances,
            'index': indices}
importances_modelo = variable_importance['importance']
indices_modelo = variable_importance['index']


# In[94]:


index = np.arange(len(names_index))
importance_desc = sorted(importances)
feature_space = []
for i in range(indices.shape[0] - 1, -1, -1):
    feature_space.append(names_index[indices[i]])
fig, ax = plt.subplots(figsize=(10, 10))
plt.title('Feature importances for Random Forest Model')
plt.barh(index,importance_desc,align="center")
plt.yticks(index,feature_space)
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature')
plt.show()


# In[95]:


conf_mat = confusion_matrix(y_teste, previsao)
sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

