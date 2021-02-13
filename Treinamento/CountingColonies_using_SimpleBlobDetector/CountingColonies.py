# Detector de Blobs

#Importando Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Importando a imagem
imagem = cv2.imread('Bolor3.png')
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

plt.imshow(imagem)
plt.show()

(thresh, imagem_tratada) = cv2.threshold(imagem, 127, 255, 0, cv2.THRESH_BINARY)
plt.imshow(imagem_tratada)
plt.show()

def create_blob_detector(roi_size=(140, 140), blob_min_area=1, 
                         blob_min_int=0, blob_max_int=.99, blob_th_step=1):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = blob_min_area
    params.maxArea = roi_size[0]*roi_size[1]
    params.filterByCircularity = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False
    # blob detection only works with "uint8" images.
    params.minThreshold = int(blob_min_int*255)
    params.maxThreshold = int(blob_max_int*255)
    params.thresholdStep = blob_th_step
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector(params)
    else:
        return cv2.SimpleBlobDetector_create(params)

detector = create_blob_detector()
saida = detector.detect(imagem_tratada)

blank = np.zeros((1,1)) 
blobs = cv2.drawKeypoints(imagem_tratada, saida, blank, (0,0,255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(saida)
text = "Numero de Colonias: " + str(len(saida))
print(text)

plt.title("Plotando as Colonias detectadas")
plt.imshow(blobs)
plt.show()
