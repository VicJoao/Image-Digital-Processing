import numpy as np
import cv2

"""
O código implementa duas técnicas de limiarização local: Niblack e Wellner. Primeiramente, converte a imagem 
colorida para escala de cinza. Em seguida, calcula a média e o desvio padrão local usando uma janela deslizante 
especificada pelo usuário. Utilizando esses valores, aplica a fórmula de Niblack e Wellner para calcular os limiares 
adaptativos. Posteriormente, aplica esses limiares na imagem para obter as imagens limiarizadas. Por fim, 
exibe as imagens originais e as imagens limiarizadas usando as técnicas de Niblack e Wellner em janelas separadas 
para fins de comparação visual.
"""

def apply_niblack_threshold(img, window_size, k):
    # Converte a imagem em escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calcula a média e o desvio padrão locais usando uma janela deslizante
    mean = cv2.blur(gray, (window_size, window_size))
    mean_square = cv2.blur(gray * gray, (window_size, window_size))
    variance = mean_square - mean * mean

    # Calcula o limiar usando a fórmula de Niblack
    threshold = mean + k * np.sqrt(variance)

    # Aplica o limiar na imagem
    thresholded_img = np.zeros_like(gray)
    thresholded_img[gray >= threshold] = 255

    return thresholded_img


def apply_wellner_threshold(img, window_size, k):
    # Converte a imagem em escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calcula a média local usando uma janela deslizante
    mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))

    # Aplica a fórmula de Wellner para calcular o limiar
    threshold = mean * (1 + k * (gray - mean) / mean)

    # Aplica o limiar na imagem
    thresholded_img = np.zeros_like(gray)
    thresholded_img[gray >= threshold] = 255

    return thresholded_img


# Carrega a imagem
image_path = 'lena.jpg'
img = cv2.imread(image_path)

# Define os parâmetros para os métodos de limiarização
window_size = 50
k_niblack = -0.9
k_wellner = 0.5

# Aplica a limiarização de Niblack
niblack_img = apply_niblack_threshold(img, window_size, k_niblack)

# Aplica a limiarização de Wellner
wellner_img = apply_wellner_threshold(img, window_size, k_wellner)

# Mostra as imagens originais e as imagens limiarizadas
cv2.imshow('Original Image', img)
cv2.imshow('Niblack Thresholding', niblack_img)
cv2.imshow('Wellner Thresholding', wellner_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
