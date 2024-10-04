import cv2
from matplotlib import pyplot as plt

"""
Este código permite ajustar o brilho e o contraste de uma imagem em tempo real usando trackbars. Ele lê uma imagem 
do disco, cria duas janelas para exibir a imagem original e a imagem com os ajustes de brilho e contraste, 
respectivamente. Dois trackbars são utilizados para controlar o brilho e o contraste da imagem ajustada. O usuário 
pode mover os trackbars para ajustar os parâmetros e visualizar imediatamente o efeito na imagem ajustada. O loop 
principal espera até que a tecla 'Esc' seja pressionada para encerrar o programa.
"""


# Define uma função vazia (callback) para os trackbars
def nothing(x):
    pass


# Lê uma imagem do disco
img = cv2.imread('wiki.png')

# Cria uma janela para a imagem
cv2.namedWindow('Imagem Original')
cv2.namedWindow('Imagem Ajustada', cv2.WINDOW_NORMAL)

# Define o tamanho da janela
cv2.resizeWindow('Imagem Ajustada', 640, 480)

# Cria dois trackbars para brilho e contraste
cv2.createTrackbar('Brilho', 'Imagem Ajustada', 0, 100, nothing)
cv2.createTrackbar('Contraste', 'Imagem Ajustada', 0, 100, nothing)

# Mostra a imagem original na janela
cv2.imshow('Imagem Original', img)

# Loop principal
while True:
    # Lê o valor dos trackbars
    brilho = cv2.getTrackbarPos('Brilho', 'Imagem Ajustada')
    contraste = cv2.getTrackbarPos('Contraste', 'Imagem Ajustada')

    # Calcula a matriz de transformação para ajustar o brilho e o contraste
    alpha = (100 + contraste) / 100
    beta = brilho

    # Aplica a transformação na imagem
    nova_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Mostra a nova imagem com brilho e contraste ajustados
    cv2.imshow('Imagem Ajustada', nova_img)

    # Espera por uma tecla ser pressionada
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Fecha todas as janelas
cv2.destroyAllWindows()
