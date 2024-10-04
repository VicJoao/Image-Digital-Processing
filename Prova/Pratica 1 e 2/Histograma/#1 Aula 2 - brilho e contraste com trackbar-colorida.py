"""
Este código Python usa a biblioteca OpenCV para calcular e exibir o histograma de uma imagem. Primeiro,
a função histograma separa a imagem em três canais de cores (B, G e R), estabelece o número de bins (256),
define os intervalos para cada canal e calcula os histogramas para cada canal de cor. Em seguida, desenha os
histogramas para os canais B, G e R em uma imagem em branco. A função controller é responsável por ajustar o brilho
e o contraste da imagem. Ela recebe os parâmetros de brilho e contraste, aplica as transformações necessárias na
imagem e retorna a imagem resultante. A função BrilhoContraste é usada como callback para os trackbars (controles
deslizantes) que controlam o brilho e o contraste da imagem. No main, a imagem é carregada, uma janela é criada
para exibi-la e dois trackbars são criados para ajustar o brilho e o contraste. A função BrilhoContraste é chamada
para exibir a imagem original e o histograma correspondente.
"""

import cv2
import numpy as np


# Função para calcular e exibir o histograma de uma imagem
def calcular_histograma(src):
    # Separa a imagem em canais de cores (B, G e R)
    bgr_planes = cv2.split(src)

    # Define o número de bins para o histograma
    histSize = 256

    # Define os intervalos para cada canal de cor
    histRange = (0, 256)  # O limite superior é exclusivo

    # Calcula os histogramas para cada canal de cor
    b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=False)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=False)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=False)

    # Desenha os histogramas para os canais B, G e R
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / histSize))

    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    # Normaliza os resultados para o intervalo (0, histImage.rows)
    cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

    # Desenha os histogramas para cada canal
    for i in range(1, histSize):
        cv2.line(histImage, (bin_w * (i - 1), hist_h - int(b_hist[i - 1])),
                 (bin_w * i, hist_h - int(b_hist[i])), (255, 0, 0), thickness=2)
        cv2.line(histImage, (bin_w * (i - 1), hist_h - int(g_hist[i - 1])),
                 (bin_w * i, hist_h - int(g_hist[i])), (0, 255, 0), thickness=2)
        cv2.line(histImage, (bin_w * (i - 1), hist_h - int(r_hist[i - 1])),
                 (bin_w * i, hist_h - int(r_hist[i])), (0, 0, 255), thickness=2)

    # Exibe a imagem com o histograma
    cv2.imshow('Histograma', histImage)


# Função para ajustar o brilho e o contraste da imagem
def ajustar_brilho_contraste(brilho=0):
    # Obtém o valor atual do brilho
    brilho = cv2.getTrackbarPos('Brilho', 'Janela')

    # Obtém o valor atual do contraste
    contraste = cv2.getTrackbarPos('Contraste', 'Janela')

    # Aplica os ajustes de brilho e contraste na imagem
    efeito = controlador(img, brilho, contraste)

    # Exibe o efeito na imagem
    cv2.imshow('Efeito', efeito)

    # Calcula e exibe o histograma do efeito
    calcular_histograma(efeito)


# Função para aplicar os ajustes de brilho e contraste na imagem
def controlador(img, brilho=255, contraste=127):
    brilho = int((brilho - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contraste = int((contraste - 0) * (127 - (-127)) / (254 - 0) + (-127))

    # Ajusta o brilho
    if brilho != 0:
        if brilho > 0:
            sombra = brilho
            maximo = 255
        else:
            sombra = 0
            maximo = 255 + brilho
        alfa = (maximo - sombra) / 255
        gama = sombra
        cal = cv2.addWeighted(img, alfa, img, 0, gama)
    else:
        cal = img

    # Ajusta o contraste
    if contraste != 0:
        alfa = float(131 * (contraste + 127)) / (127 * (131 - contraste))
        gama = 127 * (1 - alfa)
        cal = cv2.addWeighted(cal, alfa, cal, 0, gama)

    return cal


if __name__ == '__main__':
    # Carrega a imagem
    original = cv2.imread("lena.jpg")
    img = original.copy()

    # Cria uma janela para exibir a imagem
    cv2.namedWindow('Janela')
    cv2.imshow('Janela', original)

    # Cria trackbars para ajustar o brilho e o contraste
    cv2.createTrackbar('Brilho', 'Janela', 255, 2 * 255, ajustar_brilho_contraste)
    cv2.createTrackbar('Contraste', 'Janela', 127, 2 * 127, ajustar_brilho_contraste)

    # Chama a função para exibir a imagem original
    ajustar_brilho_contraste(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
