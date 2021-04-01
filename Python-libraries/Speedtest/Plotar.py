import pandas as pd
import matplotlib.pyplot as plt

arquivo = pd.read_csv('tabela.csv')

plt.plot(arquivo.Download)
plt.plot(arquivo.Upload)
plt.show()
