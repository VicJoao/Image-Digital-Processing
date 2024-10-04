import cv2
from matplotlib import pyplot as plt

# Carrega a imagem em tons de cinza
gray_img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Obtém as dimensões da imagem
altura, largura = gray_img.shape
print("Altura = ", altura, " Largura = ", largura)

# Exibe a imagem em tons de cinza
cv2.imshow('Imagem', gray_img)

# Calcula o histograma da imagem
hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

# Plota o histograma usando a biblioteca matplotlib
plt.hist(gray_img.ravel(), 256, [0, 256])
plt.title('Histograma para uma imagem em tons de cinza')
plt.show()

# Espera pela tecla ESC para sair
while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break  # Tecla ESC para sair

cv2.destroyAllWindows()
