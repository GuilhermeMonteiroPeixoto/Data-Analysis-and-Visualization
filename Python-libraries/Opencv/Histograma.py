import cv2 as cv
import matplotlib.pyplot as plt

img_rgb = cv.imread("Imagem.png")
img_grayscale = cv.imread("Imagem.png", 0)

# cv.imshow("Image", img_grayscale)

plt.figure()

colors = ('r', 'g', 'b')
for i, col in enumerate(colors):
    hist = cv.calcHist([img_rgb], [i], None, [256], [0,256])
    plt.plot(hist, color=col)
    plt.title("Histogram RGB")
    plt.xlabel("Bins")
    plt.ylabel("# of pixels")
    plt.xlim([1, 255])
plt.show()

hist_grayscale = cv.calcHist([img_grayscale], [0], None, [256], [0,256])
plt.plot(hist_grayscale)
plt.title("Histogram Gray Scale")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.xlim([1, 255])
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
