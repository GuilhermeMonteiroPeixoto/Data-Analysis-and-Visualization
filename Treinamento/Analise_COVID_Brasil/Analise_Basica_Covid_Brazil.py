import pandas as pd
import matplotlib.pyplot as plt
import folium
import numpy as np
import seaborn as sns

url= 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-total.csv'
arquivo = pd.read_csv(url, sep=',')
arquivo

df_brasil = arquivo[['state','totalCases','deaths','recovered','newCases','newDeaths','date']]

df_brasil = df_brasil.drop([0])

df_brasil.rename(columns = {'state':'Estado', 'totalCases': 'CasosTotais', 'deaths': 'Mortes', 'recovered': 'Recuperados','newCases':'NovosCasos','newDeaths':'NovasMortes','date':'Data'}, inplace = True)

df_brasil.style.background_gradient(cmap='BuPu')

sns.jointplot(x=df_brasil['CasosTotais'], y=df_brasil['Mortes'], kind="hex")

df_brasil[['Mortes','Estado']].plot.bar(x='Estado')
plt.show()

df_brasil[['NovosCasos','Estado']].plot.bar(x='Estado')
plt.show()

df_brasil[['NovasMortes','Estado']].plot.bar(x='Estado')
plt.show()

arquivo_json = 'https://raw.githubusercontent.com/luizpedone/municipal-brazilian-geodata/master/data/Brasil.json'

m = folium.Map(location=[-12.0,-49.5], zoom_start=4)

folium.Choropleth(
    geo_data=arquivo_json,
    data=df_brasil,
    columns=['Estado','Mortes'],
    key_on='feature.properties.UF',
    fill_color='RdPu',
    fill_opacity=0.9,
    legend_name=('Numero de Mortes por COVID19'),
    bins = 9,
    highlight=True
).add_to(m)

folium.LayerControl().add_to(m)

m.save('Mortes_por_COVID19.html')

m = folium.Map(location=[-12.0,-49.5], zoom_start=4)

folium.Choropleth(
    geo_data=arquivo_json,
    data=df_brasil,
    columns=['Estado','CasosTotais'],
    key_on='feature.properties.UF',
    fill_color='RdPu',
    fill_opacity=0.9,
    legend_name=('Numero de Infectados por COVID19'),
    bins = 9,
    highlight=True
).add_to(m)

folium.LayerControl().add_to(m)

m.save('Infectados_por_COVID19.html')

m = folium.Map(location=[-12.0,-49.5], zoom_start=4)

folium.Choropleth(
    geo_data=arquivo_json,
    data=df_brasil,
    columns=['Estado','Recuperados'],
    key_on='feature.properties.UF',
    fill_color='RdPu',
    fill_opacity=0.9,
    legend_name=('Numero de Recuperados'),
    bins = 9,
    highlight=True
).add_to(m)

folium.LayerControl().add_to(m)

m.save('Recuperados.html')

nordeste = df_brasil.query('Estado==("AL","BA","CE","MA","PB","PE","PI","RN","SE")')

nordeste.style.background_gradient(cmap='OrRd')

nordeste[['Mortes','Estado']].plot.bar(x='Estado')
plt.title("Nordeste")
plt.show()

nordeste[['CasosTotais','Estado']].plot.bar(x='Estado')
plt.title("Nordeste")
plt.show()

sudeste = df_brasil.query('Estado==("SP","MG","RJ","ES")')

sudeste.style.background_gradient(cmap='OrRd')

sudeste[['Mortes','Estado']].plot.bar(x='Estado')
plt.title("Sudeste")
plt.show()

sudeste[['CasosTotais','Estado']].plot.bar(x='Estado')
plt.title("Sudeste")
plt.show()
