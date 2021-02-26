import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

df = pd.read_csv('cadastro.csv')

plt.figure(figsize=(15,6))
plt.scatter(x=df['Altura'], y=df['Peso'], color='crimson', alpha=0.5)
plt.title('Altura/ Peso', weight='bold', fontsize=16)
plt.xlabel('Altura(m)', weight='bold', fontsize=12)
plt.ylabel('Peso(kg)', weight='bold', fontsize=12)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()

# Definindo variáveis
grouped_genero = df[['Altura','Peso','Idade']].groupby(df['Genero'])

aux = np.array(df['Genero'].unique())
aux = sorted(aux)

media_peso = np.array(grouped_genero['Peso'].mean())
media_idade = np.array(grouped_genero['Idade'].mean())

# Criando um gráfico
plt.bar(aux, media_peso , label = 'Peso Médio', color = 'r')
plt.bar(aux, media_idade , label = 'Idade Média', color = 'b')
plt.legend(loc='lower left')
plt.show()
