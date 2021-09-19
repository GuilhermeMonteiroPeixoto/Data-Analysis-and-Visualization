import cv2 as cv
import numpy as np

image = cv.imread('Imagem_gato1.png')
image2 = cv.imread('Imagem_gato2.png')
image3 = cv.imread('Imagem_gato3.png')
image4 = cv.imread('Imagem_cachorro1.png')

image = cv.resize(image, (200,200), interpolation = cv.INTER_AREA)
image2 = cv.resize(image2, (200,200), interpolation = cv.INTER_AREA)
image3 = cv.resize(image3, (200,200), interpolation = cv.INTER_AREA)
image4 = cv.resize(image4, (200,200), interpolation = cv.INTER_AREA)

horizontal = np.hstack((image, image2, image3, image4))

cv.imshow("Imagens", horizontal)

cv.waitKey(0)
cv.destroyAllWindows()

gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
histogram = cv.calcHist([gray_image], [0], None, [256], [0, 256])

gray_image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
histogram2 = cv.calcHist([gray_image2], [0], None, [256], [0, 256])

gray_image3 = cv.cvtColor(image3, cv.COLOR_BGR2GRAY)
histogram3 = cv.calcHist([gray_image3], [0], None, [256], [0, 256])

gray_image4 = cv.cvtColor(image4, cv.COLOR_BGR2GRAY)
histogram4 = cv.calcHist([gray_image4], [0], None, [256], [0, 256])

c1, c2, c3 = 0, 0, 0

# Euclidean Distance
i = 0
while i<len(histogram) and i<len(histogram2):
    c1+=(histogram[i]-histogram2[i])**2
    i+= 1
c1 = c1**(1 / 2)

# Euclidean Distance
i = 0
while i<len(histogram) and i<len(histogram3):
    c2+=(histogram[i]-histogram3[i])**2
    i+= 1
c2 = c2**(1 / 2)

# Euclidean Distance
i = 0
while i<len(histogram) and i<len(histogram4):
    c3+=(histogram[i]-histogram4[i])**2
    i+= 1
c3 = c3**(1 / 2)

valores = [c1,c2,c3]

for x in valores:
    print(x)

max_value = max(valores)
print('Max value:', max_value)

min_value = min(valores)
print('Min value:', min_value)
  
