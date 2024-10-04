from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np

# Carrega a imagem
src = cv.imread("lena.jpg")
if src is None:
    print('Não foi possível abrir ou encontrar a imagem')
    exit(0)

# Separa a imagem em seus canais de cor (B, G e R)
bgr_planes = cv.split(src)

# Estabelece o número de bins para o histograma
histSize = 256

# Define os intervalos de cor para os canais (B, G, R)
histRange = (0, 256)  # O limite superior é exclusivo

# Define os parâmetros do histograma
accumulate = False

# Calcula os histogramas para cada canal de cor (B, G, R)
b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

# Desenha os histogramas para cada canal (B, G, R)
hist_w = 512
hist_h = 400
bin_w = int(round(hist_w / histSize))
histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

# Normaliza os histogramas para o intervalo (0, histImage.rows)
cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

# Desenha as linhas do histograma para cada canal (B, G, R)
for i in range(1, histSize):
    cv.line(histImage, (bin_w*(i-1), hist_h - int(b_hist[i-1])),
            (bin_w*i, hist_h - int(b_hist[i])), (255, 0, 0), thickness=2)
    cv.line(histImage, (bin_w*(i-1), hist_h - int(g_hist[i-1])),
            (bin_w*i, hist_h - int(g_hist[i])), (0, 255, 0), thickness=2)
    cv.line(histImage, (bin_w*(i-1), hist_h - int(r_hist[i-1])),
            (bin_w*i, hist_h - int(r_hist[i])), (0, 0, 255), thickness=2)

# Exibe a imagem original e o histograma
cv.imshow('Imagem Original', src)
cv.imshow('Histograma', histImage)
cv.waitKey()
cv.destroyAllWindows()
