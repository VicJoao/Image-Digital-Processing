from skimage.filters import threshold_local
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Este código aplica limiarização adaptativa em uma imagem utilizando tanto o OpenCV quanto a biblioteca 
scikit-image. Aqui está uma explicação detalhada do que cada parte do código faz:

Carrega uma imagem ("sudoku.png") em tons de cinza. Calcula e exibe o histograma da imagem usando a função 
np.histogram da biblioteca NumPy. Aplica limiarização adaptativa usando OpenCV com o método de limiarização médio 
adaptativo (cv2.ADAPTIVE_THRESH_MEAN_C). O limiar é calculado para cada pixel com base na média dos valores dos 
pixels na vizinhança especificada (neighbourhood_size) e um valor de compensação (constant_c). O resultado é uma 
imagem binarizada onde os pixels acima do limiar são definidos como preto e os pixels abaixo do limiar são definidos 
como branco (devido ao uso de cv2.THRESH_BINARY_INV). Aplica limiarização adaptativa usando scikit-image com o método 
threshold_local. Neste método, o limiar é calculado para cada pixel com base na vizinhança especificada ( 
neighbourhood_size) e um valor de compensação (constant_c). A imagem resultante é convertida para a faixa de 8 bits 
usando np.uint8 e multiplicada por 255 para garantir que os pixels sejam preto (0) ou branco (255).
"""

# Carrega a imagem e converte para escala de cinza
img = cv2.imread("sudoku.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Exibe o histograma da imagem
hist, bins = np.histogram(gray.ravel(), 256, [0, 256])
plt.plot(hist)
plt.title("Histograma usando NumPy")
plt.show()

# Aplica limiarização adaptativa com OpenCV
neighbourhood_size = 25
constant_c = 15
thresh_opencv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY_INV, neighbourhood_size, constant_c)
cv2.imshow("Limiar Adaptativo OpenCV", thresh_opencv)

# Aplica limiarização adaptativa com scikit-image
neighbourhood_size = 29
constant_c = 5
threshold_value = threshold_local(gray, neighbourhood_size, offset=constant_c)
thresh_scikit = (gray < threshold_value).astype(np.uint8) * 255
cv2.imshow("Limiar Adaptativo scikit-image", thresh_scikit)

cv2.waitKey(0)
