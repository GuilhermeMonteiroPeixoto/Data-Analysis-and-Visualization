#AULA 2 OPENCV

#Importando Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Defininco metodo createBackgroundSubtractor
BackgroundSub = cv2.createBackgroundSubtractorMOG2()

imagem1 = cv2.imread('referencia.jpg')
imagem2 = cv2.imread('atual.png')
fgMask1 = BackgroundSub.apply(imagem1)
fgMask2 = BackgroundSub.apply(imagem2)

plt.imshow(cv2.cvtColor(fgMask2, cv2.COLOR_BGR2RGB))
plt.show()
