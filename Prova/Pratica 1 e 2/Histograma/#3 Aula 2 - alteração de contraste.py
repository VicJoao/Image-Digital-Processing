"""
O código processa uma imagem (no caso, a imagem 'lena.png') para ajustar o brilho e o contraste. Inicialmente,
a imagem é convertida para escala de cinza usando cv2.cvtColor(). Em seguida, define-se os valores desejados de
aumento de brilho (alpha) e contraste (beta). Utilizando a função cv2.addWeighted(), aplica-se o ajuste de brilho e
contraste à imagem em escala de cinza. A imagem resultante é exibida em uma janela. Além disso, o código calcula os
valores mínimos e máximos dos píxeis da imagem original usando cv2.minMaxLoc(). Em seguida, aplica-se um ajuste
bilinear de contraste utilizando cv2.convertScaleAbs() para obter um contraste desejado. Por fim, a imagem resultante
é exibida em uma janela novamente.
"""


import cv2
import numpy as np

# Carrega a imagem
imagem = cv2.imread('lena.png')

# Converte a imagem para escala de cinza
imagem_grayscale = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Define os valores de brilho e contraste desejados
alpha = 1.5  # Aumento de 50% no brilho
beta = 30   # Aumento de 30 unidades no contraste

# Aplica o ajuste de brilho e contraste
imagem_ajustada = cv2.addWeighted(imagem_grayscale, alpha, np.zeros(imagem_grayscale.shape, dtype=imagem_grayscale.dtype), 0, beta)

# Exibe a imagem resultante
cv2.imshow("Imagem Ajustada", imagem_ajustada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Define o valor de contraste desejado
contraste = 30  # Aumento de 30 unidades no contraste

# Calcula os valores mínimos e máximos dos pixels da imagem
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imagem_grayscale)
print(f"Valor Mínimo: {minVal}, Valor Máximo: {maxVal}")

# Aplica o ajuste bilinear do contraste
imagem_ajustada = cv2.convertScaleAbs(imagem_grayscale, alpha=(255.0/(maxVal-minVal)), beta=(-255.0*minVal/(maxVal-minVal))+255*contraste/100)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imagem_ajustada)
print(f"Novo Valor Mínimo: {minVal}, Novo Valor Máximo: {maxVal}")

# Exibe a imagem resultante após o ajuste de contraste
cv2.imshow("Imagem Ajustada", imagem_ajustada)
cv2.waitKey(0)
cv2.destroyAllWindows()
