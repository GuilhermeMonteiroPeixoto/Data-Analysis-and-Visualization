import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_fusarium = pd.read_csv('fusarium.csv')

locations = data_fusarium[['Latitude', 'Longitude']]
locationlist = locations.values.tolist()
len(locationlist)

import folium
map = folium.Map(location=[13.2,29.5], zoom_start=2)
for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point], popup=data_fusarium['Region'][point]).add_to(map)

folium.LayerControl().add_to(map)
map.save('index.html')
