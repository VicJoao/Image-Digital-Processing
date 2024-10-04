import cv2

"""código dado carrega uma imagem em tons de cinza e aplica a equalização de histograma global e local utilizando a 
função equalizeHist e o algoritmo CLAHE (Contrast Limited Adaptive Histogram Equalization), respectivamente. O CLAHE 
é utilizado para melhorar o contraste localmente na imagem, evitando a saturação do contraste em áreas de alta 
variação."""

# Carrega a imagem em tons de cinza
gray_img = cv2.imread('estatua.png', 0)
equ = cv2.equalizeHist(gray_img)

# Define o tamanho da janela de equalização local
win_size = (32, 32)

# Aplica a equalização local do histograma
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=win_size)
equalized_img = clahe.apply(gray_img)

# Mostra a imagem original e a imagem equalizada
cv2.imshow('Imagem original', gray_img)
cv2.imshow('Imagem equalizada', equ)
cv2.imshow('Imagem Equalizada com clahe', equalized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
