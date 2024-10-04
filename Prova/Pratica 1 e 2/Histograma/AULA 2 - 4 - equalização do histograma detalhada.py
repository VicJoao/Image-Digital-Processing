import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
Este código carrega uma imagem em tons de cinza ('wiki.png') e equaliza essa imagem usando cv2.equalizeHist(). Em 
seguida, ele calcula e plota o histograma e o CDF (Função de Distribuição Cumulativa) tanto para a imagem original 
quanto para a imagem equalizada. Após isso, ele exibe a imagem original e a imagem equalizada em dois subplots 
separados. O histograma mostra a distribuição das intensidades de pixel na imagem, enquanto o CDF mostra a acumulação 
das intensidades de pixel ao longo do intervalo. Essas visualizações ajudam a compreender como a equalização do 
histograma afeta a distribuição das intensidades de pixel na imagem.
"""

# Carrega a imagem em tons de cinza
img = cv2.imread('wiki.png', 0)

# Equaliza a imagem
equ = cv2.equalizeHist(img)

# Calcula e plota o histograma e o CDF da imagem original
plt.subplot(221)
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('CDF', 'Histograma'), loc='upper left')
plt.title('Histograma da Imagem Original')
plt.xticks([]), plt.yticks([])

# Calcula e plota o histograma e o CDF da imagem equalizada
plt.subplot(222)
hist, bins = np.histogram(equ.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
plt.plot(cdf_normalized, color='b')
plt.hist(equ.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('CDF', 'Histograma'), loc='upper left')
plt.title('Histograma da Imagem Equalizada')
plt.xticks([]), plt.yticks([])

# Exibe a imagem original
plt.subplot(223)
plt.imshow(img, cmap='gray')
plt.title('Imagem Original')
plt.xticks([]), plt.yticks([])

# Exibe a imagem equalizada
plt.subplot(224)
plt.imshow(equ, cmap='gray')
plt.title('Imagem Equalizada')
plt.xticks([]), plt.yticks([])

plt.show()
