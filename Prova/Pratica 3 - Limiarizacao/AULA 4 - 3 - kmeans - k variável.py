import numpy as np
import cv2

"""
Este código implementa o algoritmo de agrupamento K-Means para segmentar uma imagem em regiões de cores 
semelhantes. Primeiro, a imagem é carregada e os pixels são reformatados em um array bidimensional. Em seguida, 
o algoritmo K-Means é aplicado para agrupar os pixels em um número especificado de clusters. Os centros dos clusters 
são calculados e a imagem original é reconstruída com base nos rótulos dos clusters atribuídos a cada pixel. A imagem 
resultante é exibida, mostrando a imagem original e a versão segmentada. Além disso, os centros dos clusters e as 
cores predominantes de cada cluster são impressos. Cada cor predominante é exibida em uma imagem separada, 
permitindo visualizar a aparência dos clusters individuais.
"""

# Carrega a imagem
img = cv2.imread('amazonia1.jpg')

# Reshape para um array 2D
Z = img.reshape((-1, 3))
Z = np.float32(Z)

# Define os critérios, número de clusters (K) e aplica o kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = int(input("Entre com o número de clusters (2 a 10): "))
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Converte de volta para uint8 e recria a imagem original
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape(img.shape)
final = np.concatenate((img, res2), axis=1)
cv2.imshow('res2', final)

# Exibe os centros dos clusters
num = center.shape[0]
for j in range(num):
    print("Centro do cluster %d:" % j)
    print(center[j])

# Cria imagens com a cor de cada cluster
k = np.ones((10, 100, 100, 3), np.uint8)
for j in range(num):
    k[j, :, :, 0] = [center[j][0]]
    k[j, :, :, 1] = [center[j][1]]
    k[j, :, :, 2] = [center[j][2]]

# Exibe a cor de cada cluster em uma imagem separada
final = np.concatenate((k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]), axis=1)
cv2.imshow("Cores dos Clusters", final)

cv2.waitKey(0)
cv2.destroyAllWindows()
