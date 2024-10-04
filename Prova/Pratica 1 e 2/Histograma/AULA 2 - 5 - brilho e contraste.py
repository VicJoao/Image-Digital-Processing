import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
Este código carrega uma imagem em tons de cinza ('lena.jpg') e realiza duas operações diferentes sobre ela. 
Primeiro, ele adiciona um deslocamento de brilho de 20 para aumentar o brilho da imagem original. Em seguida, 
ele solicita ao usuário que insira valores para contraste e brilho e aplica uma transformação linear na imagem com 
base nesses valores. Depois de cada operação, ele exibe a imagem original e o histograma correspondente, bem como a 
imagem transformada e o histograma correspondente. Isso permite visualizar como as operações afetam a distribuição 
das intensidades de pixel na imagem.
"""

# Carrega a imagem em tons de cinza
image = cv.imread("lena.jpg", cv.IMREAD_GRAYSCALE)

# Calcula os valores máximo e mínimo de intensidade de pixel na imagem original
min_val = np.min(image)
max_val = np.max(image)
print("Valor máximo:", max_val, "Valor mínimo:", min_val)

# Aplica um deslocamento de 20 na imagem para aumentar o brilho
nova = (image + 20) * (255 / (max_val - min_val))
nova = np.uint8(nova)
print("Novo valor máximo:", np.max(nova), "Novo valor mínimo:", np.min(nova))

# Exibe a imagem original e a imagem após o deslocamento
plt.subplot(221), plt.imshow(image, cmap='gray')
plt.subplot(222), plt.imshow(nova, cmap='gray')
plt.subplot(223), plt.hist(image.ravel(), 256, [0, 256]), plt.title('Histograma da Imagem Original')
plt.subplot(224), plt.hist(nova.ravel(), 256, [0, 256]), plt.title('Histograma da Imagem com Deslocamento')
plt.show()

# Solicita os valores de contraste e brilho do usuário
new_image = np.zeros(image.shape, image.dtype)
alpha = 1.0  # Controle de contraste
beta = 0    # Controle de brilho

print('Transformação linear básica:')
print('-------------------------')
try:
    alpha = float(input('* Entre com o valor de alpha (contraste) [1.0-3.0]: '))
    beta = int(input('* Entre com o valor de beta (brilho) [0-100]: '))
except ValueError:
    print('Erro, valor inválido')

# Aplica a transformação linear na imagem
new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)

# Exibe a imagem original e a imagem após a transformação linear
plt.subplot(221), plt.imshow(image, cmap='gray')
plt.subplot(222), plt.imshow(new_image, cmap='gray')

plt.subplot(223), plt.hist(image.ravel(), 256, [0, 256]), plt.title('Histograma da Imagem Original')
plt.subplot(224), plt.hist(new_image.ravel(), 256, [0, 256]), plt.title('Histograma da Imagem com Transformação Linear')

plt.show()
