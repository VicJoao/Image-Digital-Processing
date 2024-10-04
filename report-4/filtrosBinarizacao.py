import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('aviao.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar filtro Sobel
sobel_x = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)

# Aplicar filtro Prewitt
prewitt_kernel_x = np.array([[1, 0, -1],
                              [1, 0, -1],
                              [1, 0, -1]])
prewitt_kernel_y = np.array([[1, 1, 1],
                              [0, 0, 0],
                              [-1, -1, -1]])
prewitt_x = cv2.filter2D(imagem, -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(imagem, -1, prewitt_kernel_y)
prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)

# Aplicar filtro Roberts
roberts_kernel_x = np.array([[1, 0],
                              [0, -1]])
roberts_kernel_y = np.array([[0, 1],
                              [-1, 0]])
roberts_x = cv2.filter2D(imagem, -1, roberts_kernel_x)
roberts_y = cv2.filter2D(imagem, -1, roberts_kernel_y)
roberts = np.sqrt(roberts_x**2 + roberts_y**2)

# Binarização usando Otsu
_, sobel_bin = cv2.threshold(sobel.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, prewitt_bin = cv2.threshold(prewitt.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, roberts_bin = cv2.threshold(roberts.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Exibir as imagens binarizadas
cv2.imshow('Sobel Binarizado', sobel_bin)
cv2.imshow('Prewitt Binarizado', prewitt_bin)
cv2.imshow('Roberts Binarizado', roberts_bin)

cv2.waitKey(0)
cv2.destroyAllWindows()
