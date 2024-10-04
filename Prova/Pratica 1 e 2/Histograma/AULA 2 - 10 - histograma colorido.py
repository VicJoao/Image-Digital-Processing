from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np

"""
Este código em Python usando OpenCV calcula e exibe o histograma de uma imagem colorida. Primeiro, a imagem é 
carregada usando a função cv.imread. Em seguida, a imagem é dividida em três canais de cores (azul, verde e vermelho) 
usando a função cv.split.

O número de bins (níveis de intensidade) para o histograma é estabelecido como 256. Em seguida, os histogramas de 
cada canal de cor são calculados usando a função cv.calcHist, que retorna o número de pixels em cada bin.

Para desenhar os histogramas, uma imagem em branco é inicializada e normalizada para o intervalo de altura 
especificado. Cada canal de cor é normalizado separadamente e, em seguida, as linhas são desenhadas no histograma 
para cada canal de cor, com cores diferentes.

Finalmente, as imagens original e o histograma são exibidos usando cv.imshow, e a execução é interrompida até que uma 
tecla seja pressionada.

"""

# Carrega a imagem
src = cv.imread("lena.jpg")
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Separa a imagem em 3 canais (B, G e R)
bgr_planes = cv.split(src)

# Define o número de bins para o histograma
histSize = 256

# Define o intervalo para os canais (para B, G, R)
histRange = (0, 256)  # O limite superior é exclusivo

# Parâmetro para o cálculo do histograma
accumulate = False

# Calcula os histogramas para cada canal de cor
b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

# Define as dimensões da imagem do histograma
hist_w = 512
hist_h = 400
bin_w = int(round(hist_w / histSize))

# Cria uma imagem em branco para desenhar os histogramas
histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

# Normaliza os resultados para o intervalo (0, histImage.rows)
cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

# Desenha o histograma para cada canal de cor
for i in range(1, histSize):
    cv.line(histImage, (bin_w * (i - 1), hist_h - int((b_hist[i - 1][0]))),
            (bin_w * (i), hist_h - int((b_hist[i][0]))), (255, 0, 0), thickness=2)

    cv.line(histImage, (bin_w * (i - 1), hist_h - int((g_hist[i - 1][0]))),
            (bin_w * (i), hist_h - int((g_hist[i][0]))), (0, 255, 0), thickness=2)

    cv.line(histImage, (bin_w * (i - 1), hist_h - int((r_hist[i - 1][0]))),
            (bin_w * (i), hist_h - int((r_hist[i][0]))), (0, 0, 255), thickness=2)


# Exibe a imagem original e o histograma
cv.imshow('Source image', src)
cv.imshow('calcHist Demo', histImage)

# Aguarda até que uma tecla seja pressionada
cv.waitKey()

# Fecha todas as janelas
cv.destroyAllWindows()
