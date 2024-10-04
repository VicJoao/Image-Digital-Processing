import cv2
import matplotlib.pyplot as plt

# Carregar a imagem
image = cv2.imread('image.png', 0)  # Carregar como escala de cinza

# Inicializar o CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Aplicar o CLAHE na imagem
equalized_image = clahe.apply(image)

# Calcular histograma da imagem original
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])

# Calcular histograma da imagem equalizada
hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

# Plotar os histogramas
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(hist_original, color='black')
plt.title('Histograma da Imagem Original')
plt.xlabel('Níveis de Cinza')
plt.ylabel('Número de Pixels')

plt.subplot(1, 2, 2)
plt.plot(hist_equalized, color='black')
plt.title('Histograma da Imagem Equalizada')
plt.xlabel('Níveis de Cinza')
plt.ylabel('Número de Pixels')

plt.tight_layout()

# Mostrar a imagem original e a imagem equalizada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagem Original')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Imagem Equalizada usando CLAHE')

plt.tight_layout()
plt.show()
