import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
O código carrega uma imagem em tons de cinza, em seguida, aplica um filtro de mediana para suavizar a imagem. Em 
seguida, ele realiza três tipos diferentes de limiarização: limiarização global com um valor de limiar de 127, 
limiarização adaptativa usando o método da média e limiarização adaptativa usando o método gaussiano. As imagens 
resultantes de cada método são exibidas em uma grade junto com seus respectivos títulos. Esse processo permite 
comparar visualmente os efeitos de diferentes técnicas de limiarização na imagem original, facilitando a escolha da 
melhor abordagem para segmentar objetos de interesse em uma imagem.
"""

# Carrega a imagem em tons de cinza
img = cv.imread('sudoku.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "O arquivo não pôde ser lido, verifique com os.path.exists()"
img = cv.medianBlur(img, 5)

# Aplica limiarização global
ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# Aplica limiarização adaptativa usando o método da média
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY, 11, 2)

# Aplica limiarização adaptativa usando o método gaussiano
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv.THRESH_BINARY, 11, 2)

# Prepara os títulos para as imagens
titles = ['Imagem Original', 'Limiarização Global (v = 127)',
          'Limiarização Adaptativa Média', 'Limiarização Adaptativa Gaussiana']

# Prepara as imagens para exibição
images = [img, th1, th2, th3]

# Exibe as imagens e seus títulos
for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
