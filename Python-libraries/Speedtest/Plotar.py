import pandas as pd
import matplotlib.pyplot as plt

arquivo = pd.read_csv('tabela.csv')

plt.plot(arquivo.Download)
plt.plot(arquivo.Upload)
plt.legend(["Download", "Upload"], loc =0)

plt.title('Média Download: {} e Média Upload: {}'.format(round(arquivo['Download'].mean(),2), round(arquivo['Upload'].mean(),2)))
plt.xlabel('time (min)',fontsize=14)
plt.ylabel('Mbps',fontsize=14)

plt.show()
