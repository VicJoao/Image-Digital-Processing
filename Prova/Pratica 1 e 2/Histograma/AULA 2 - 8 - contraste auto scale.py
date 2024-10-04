import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
Este código primeiro carrega a imagem em tons de cinza e, em seguida, calcula os valores mínimo e máximo da 
intensidade dos pixels na imagem original. Em seguida, aplica uma transformação linear para ajustar o brilho da 
imagem. A transformação é aplicada pixel a pixel, e os valores resultantes são armazenados em uma nova matriz. Após 
isso, calcula novamente os valores mínimo e máximo da intensidade dos pixels na imagem ajustada. Por fim, 
exibe a imagem original e a imagem ajustada, juntamente com seus histogramas para comparação.

A transformação linear aplicada neste código ajusta o brilho da imagem em tons de cinza. 
"""

# Carrega a imagem em tons de cinza
image = cv.imread("wiki.png", cv.IMREAD_GRAYSCALE)

# Imprime as dimensões da imagem
print("Dimensões da imagem:", image.shape)

# Calcula os valores mínimo e máximo da intensidade dos pixels na imagem original
min_val = np.min(image)
max_val = np.max(image)
print("Valor Mínimo:", min_val, "Valor Máximo:", max_val)

# Aplica a transformação linear na imagem para ajustar o brilho
new_image = np.zeros(image.shape, dtype=image.dtype)
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        new_image[y, x] = np.uint8(10 + 255 * ((image[y, x] - min_val) / (max_val - min_val)))

# Calcula os novos valores mínimo e máximo da intensidade dos pixels na imagem ajustada
min_val_new = np.min(new_image)
max_val_new = np.max(new_image)
print("Novo Valor Mínimo:", min_val_new, "Novo Valor Máximo:", max_val_new)

# Exibe a imagem original e a imagem com transformação linear
plt.subplot(221), plt.imshow(image, cmap='gray')
plt.subplot(222), plt.imshow(new_image, cmap='gray')

# Exibe o histograma da imagem original e da imagem com transformação linear
plt.subplot(223), plt.hist(image.ravel(), 256, [0, 256]), plt.title('Histograma da Imagem Original')
plt.subplot(224), plt.hist(new_image.ravel(), 256, [0, 256]), plt.title('Histograma após Transformação Linear')

plt.show()
