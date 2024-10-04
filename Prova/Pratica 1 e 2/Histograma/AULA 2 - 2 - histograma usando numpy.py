import cv2
import matplotlib.pyplot as plt
import numpy as np

# Carrega a imagem 'lena.jpg'
img = cv2.imread('lena.jpg', 1)

# Calcula o histograma da imagem usando numpy
hist, bins = np.histogram(img.ravel(), 256, [0, 256])

# Plota o histograma
plt.plot(hist)
plt.title("Histograma usando NumPy")
plt.xlabel("Intensidade de pixel")
plt.ylabel("Frequência")
plt.show()

"""
Eixo X (Intensidade de Píxel): O eixo horizontal representa os valores de intensidade de píxel, que geralmente 
variam de 0 a 255 para imagens em escala de cinza. Para imagens coloridas, existem histogramas separados para cada 
canal de cor (vermelho, verde, azul), e os valores de intensidade geralmente variam de 0 a 255 para cada canal.

Eixo Y (Frequência): O eixo vertical representa a frequência ou quantidade de píxeis com uma determinada 
intensidade. Quanto mais alto o valor em um determinado ponto do eixo X, mais píxeis na imagem têm essa intensidade.
"""
