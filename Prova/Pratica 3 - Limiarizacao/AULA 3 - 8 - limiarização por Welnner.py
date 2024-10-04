import cv2
import numpy as np

"""
O código implementa o algoritmo de limiarização adaptativa de Wellner em uma imagem em escala de cinza. Ele 
percorre a imagem pixel por pixel, calculando um limiar adaptativo com base na média dos valores de intensidade na 
vizinhança do pixel atual. A vizinhança é definida por uma janela de tamanho determinado pelo parâmetro window_size. 
Para cada pixel, a média local é calculada na região de interesse ao redor dele e usada como limiar para binarizar o 
pixel. O resultado é uma imagem binarizada, onde os pixels são classificados como preto ou branco dependendo se sua 
intensidade é maior ou menor que o limiar adaptativo calculado. Este método é eficaz para segmentar imagens com 
variações locais de iluminação.
"""


def wellner_adaptive_threshold(img, window_size=15):
    # Calcula o tamanho da imagem
    height, width = img.shape

    # Inicializa a imagem binarizada
    binary_img = np.zeros_like(img, dtype=np.uint8)

    # Loop sobre a imagem para calcular o limiar adaptativo
    for y in range(height):
        for x in range(width):
            # Define a região de interesse ao redor do pixel atual
            y1 = max(0, y - window_size // 2)
            y2 = min(height, y + window_size // 2 + 1)
            x1 = max(0, x - window_size // 2)
            x2 = min(width, x + window_size // 2 + 1)
            roi = img[y1:y2, x1:x2]
            # Calcula a média local
            mean = np.mean(roi)
            threshold = mean
            # Binariza o pixel de acordo com o limiar adaptativo
            if img[y, x] > threshold:
                binary_img[y, x] = 255

    return binary_img


# Carrega a imagem
img = cv2.imread('imagem1.png', cv2.IMREAD_GRAYSCALE)

# Aplica o limiar de Wellner adaptativo
thresholded_img = wellner_adaptive_threshold(img)

# Exibe a imagem original e a imagem limiarizada
cv2.imshow('Original Image', img)
cv2.imshow('Adaptive Wellner Thresholded Image', thresholded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
