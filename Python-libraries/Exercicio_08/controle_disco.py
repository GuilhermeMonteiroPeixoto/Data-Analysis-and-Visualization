import pandas as pd

data = pd.read_csv('usuario.txt', header = None)
data.columns = ['usuario','espaco_utilizado']

serie_espaco = data['espaco_utilizado'].map(lambda x: x*0.00000095367)

Espaco_total_ocupado = serie_espaco.sum()
Espaco_medio_ocupado = Espaco_total_ocupado/len(serie_espaco)

serie_porcent_uso = serie_espaco.map(lambda x: 100*x/Espaco_total_ocupado)

df = pd.concat([data['usuario'],serie_espaco,serie_porcent_uso], axis=1)
df.columns = ['usuario','espaco_utilizado','%_de_uso']
df[['espaco_utilizado','%_de_uso']].style.format('{:.02f}')
df.sort_values(['%_de_uso'],ascending=False, inplace=True)
df.to_csv('relatorio.txt')

arquivo = open('relatorio.txt','a')
arquivo.write('\nEspaço total ocupado: '+str(Espaco_total_ocupado))
arquivo.write('\nEspaço médio ocupado: '+str(Espaco_medio_ocupado))
arquivo.close()

