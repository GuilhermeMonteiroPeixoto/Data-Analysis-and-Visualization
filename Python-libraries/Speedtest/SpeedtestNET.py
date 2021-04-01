import speedtest
import pandas as pd
from datetime import datetime
from threading import Timer
import matplotlib.pyplot as plt

speed = speedtest.Speedtest()
def internet():
    Montarcsv = pd.read_csv('tabela.csv')
    
    Dspeed = speed.download()/(1050625)
    Uspeed = speed.upload()/(1050625)

    data_atual = datetime.now().strftime('%d/%m/%Y')
    hora_atual = datetime.now().strftime('%H:%M:%S')

    d = {'Data': data_atual, 'Hora': hora_atual, 'Download': Dspeed, 'Upload': Uspeed}
    Montarcsv = Montarcsv.append(d, ignore_index=True)

    Montarcsv.to_csv('tabela.csv', index=False)
    
    Timer(10,internet).start()

internet()
