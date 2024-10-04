"""O comando cv2.createTrackbar() da OpenCV é usado para criar uma barra de controle (trackbar) em uma janela de
imagem. Essa trackbar permite que o usuário ajuste um parâmetro específico em tempo real e veja imediatamente o
efeito das alterações na imagem. Cv2.createTrackbar(nome, janela, valor_inicial, valor_maximo, callback) nome: O nome
da trackbar que será exibido na janela. Janela: O nome da janela onde a trackbar será exibida. Valor_inicial: O valor
inicial do parâmetro controlado pela trackbar. Valor_maximo: O valor máximo permitido para o parâmetro. Callback: Uma
função de retorno de chamada que será chamada sempre que o valor da trackbar for alterado. Essa função recebe um
único argumento, que é o valor atual da trackbar.

"""

import cv2
import numpy as np


# Função para calcular e exibir o histograma de uma imagem em tons de cinza
def histograma(src):
    histSize = 256
    histRange = (0, 256)  # O limite superior é exclusivo
    accumulate = False

    # Calcula o histograma da imagem em tons de cinza
    grey_hist = cv2.calcHist([src], [0], None, [histSize], histRange, accumulate=accumulate)

    # Desenha o histograma
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / histSize))
    histImage = np.zeros((hist_h, hist_w, 1), dtype=np.uint8)

    cv2.normalize(grey_hist, grey_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

    for i in range(1, histSize):
        cv2.line(histImage, (bin_w * (i - 1), hist_h - int(grey_hist[i - 1])),
                 (bin_w * i, hist_h - int(grey_hist[i])), (255, 0, 0), thickness=2)

    cv2.imshow('Histograma', histImage)


# Função para ajustar brilho e contraste
def ajustar_brilho_contraste(Brilho=0):
    Brilho = cv2.getTrackbarPos('Brilho', 'CCF394')
    Contraste = cv2.getTrackbarPos('Contraste', 'CCF394')
    efeito = controller(img, Brilho, Contraste)
    cv2.imshow('Efeito', efeito)
    histograma(efeito)


# Função para aplicar os ajustes de brilho e contraste
def controller(img, Brilho=255, Contraste=127):
    Brilho = int((Brilho - 0) * (255 - (-255)) / (510 - 0) + (-255))
    Contraste = int((Contraste - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if Brilho != 0:
        if Brilho > 0:
            sombra = Brilho
            maximo = 255
        else:
            sombra = 0
            maximo = 255 + Brilho
        al_pha = (maximo - sombra) / 255
        ga_mma = sombra
        cal = cv2.addWeighted(img, al_pha, img, 0, ga_mma)
    else:
        cal = img

    if Contraste != 0:
        Alpha = float(131 * (Contraste + 127)) / (127 * (131 - Contraste))
        Gamma = 127 * (1 - Alpha)
        cal = cv2.addWeighted(cal, Alpha, cal, 0, Gamma)

    return cal


if __name__ == '__main__':
    original = cv2.imread("lena.jpg", 0)
    img = original.copy()

    cv2.namedWindow('CCF394')
    cv2.imshow('CCF394', original)

    cv2.createTrackbar('Brilho', 'CCF394', 255, 2 * 255, ajustar_brilho_contraste)
    cv2.createTrackbar('Contraste', 'CCF394', 127, 2 * 127, ajustar_brilho_contraste)

    ajustar_brilho_contraste(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
