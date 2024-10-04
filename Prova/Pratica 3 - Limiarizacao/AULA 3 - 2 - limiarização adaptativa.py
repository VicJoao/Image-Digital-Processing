import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
O código carrega uma imagem de um sudoku em escala de cinza, aplica um filtro de mediana para suavizar possíveis 
ruídos e, em seguida, exibe o histograma da imagem usando a biblioteca numpy e matplotlib. Em seguida, 
realiza a limiarização da imagem usando dois métodos diferentes: o método de limiarização global e os métodos de 
limiarização adaptativa baseados na média e na gaussiana. No método de limiarização global, um único valor de limiar 
é usado para todos os pixels da imagem, enquanto nos métodos adaptativos, o limiar é calculado localmente para cada 
região da imagem. Os resultados são exibidos lado a lado com a imagem original e as imagens binarizadas usando os 
diferentes métodos de limiarização. O método de limiarização adaptativa baseado na média calcula o limiar como a 
média ponderada dos valores dos pixels vizinhos, enquanto o método baseado na gaussiana utiliza uma média ponderada 
ponderada de forma gaussiana. Isso permite uma binarização mais precisa e adaptativa, especialmente em regiões com 
iluminação variável. As imagens resultantes são exibidas em um subplot com seus respectivos títulos, permitindo a 
comparação visual dos diferentes métodos de limiarização."""


img = cv.imread('sudoku.png', cv.IMREAD_GRAYSCALE)
img = cv.medianBlur(img, 5)
img = cv.imread("sudoku.png")
assert img is not None, "file could not be read, check with os.path.exists()"

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("entrada", gray)

hist, bins = np.histogram(gray.ravel(), 256, [0, 256])
plt.plot(hist)
plt.title("histograma usando Numpy")
plt.show()

'''
pelo comando são:

th2: A imagem binarizada resultante. 255: Valor máximo do pixel para a imagem de saída (valor branco). 
cv.ADAPTIVE_THRESH_MEAN_C: Método de limiarização adaptativa que utiliza a média ponderada dos blocos vizinhos para 
determinar o limiar. cv.THRESH_BINARY: Tipo de limiarização binária, onde os pixels são classificados como preto ou 
branco. 11: Tamanho do bloco utilizado para calcular a média ponderada dos pixels vizinhos. 2: Valor de compensação 
subtraído da média ponderada dos pixels vizinhos antes de aplicar o limiar adaptativo.

'''

ret, th1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

th2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY, 11, 2)
th3 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                           cv.THRESH_BINARY, 11, 2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
