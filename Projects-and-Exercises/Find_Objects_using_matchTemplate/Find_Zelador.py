#AULA 1 OPENCV

#Importando Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import cv2
#Lendo imagem
image = cv2.imread('simp.png')
#Set RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Plotar imagem_original
plt.subplot(1, 2, 1)
plt.title("Identificando o Zelador")
plt.imshow(image)

#Transformar imagem em gray_scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Ler imagem_rosto
template = cv2.imread('zelador.png',0)

#Compara uma imagem_modelo com regiões de imagem_original
#Usando o metodo cv2.TM_CCOEFF
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
#cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
#cv2.matchTemplate(gray, template, cv2.TM_CCORR)
#cv2.matchTemplate(gray, template, cv2.TM_SQDIFF)

#Encontre global minimum and maximum
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

#Desenhando o retangulo
top_left = max_loc
bottom_right = (top_left[0] + 50,top_left[1] + 50)
cv2.rectangle(image, top_left, bottom_right, (0,0,255), 2)

#Plotando imagem_original com retangulo
plt.subplot(1, 2, 2)
plt.title("Zelador")
plt.imshow(image)
plt.show()
