import cv2
import numpy as np

"""
Este código tem como objetivo realizar a segmentação de uma imagem RGB utilizando o algoritmo K-Means. 
Inicialmente, a imagem é carregada utilizando a função cv2.imread() e transformada em uma matriz bidimensional de 
pixels, onde cada pixel é representado por uma tupla de três valores correspondentes aos componentes de cor RGB. Em 
seguida, as tuplas únicas de pixels são extraídas e contadas para determinar quantas combinações únicas de cores 
foram utilizadas na imagem original. O usuário é então solicitado a inserir o número desejado de clusters para a 
segmentação. O algoritmo K-Means é aplicado aos pixels da imagem, onde os clusters são gerados com base na 
similaridade das cores dos pixels. Os centróides dos clusters são calculados e usados para representar as cores 
dominantes de cada cluster. Em seguida, os pixels da imagem são substituídos pelos valores dos centróides 
correspondentes, resultando em uma nova imagem segmentada. Além disso, os centróides de cada cluster são exibidos na 
saída. Por fim, as barras de cores utilizadas na imagem resultante são montadas e exibidas em uma única imagem para 
visualização.
"""

img = cv2.imread('amazonia1.jpg')
Z = img.reshape((-1, 3))
# Transformar a imagem em uma lista de tuplas (r, g, b)

tuplas_unicas = set(map(tuple, Z))

# Contar quantas tuplas únicas foram encontradas
num_cor_pixel_utilizada = len(tuplas_unicas)
print("numero de tuplas unicas encontradas: %s" % num_cor_pixel_utilizada)
print("Esta imagem utilizou %d combinações de cores RGB" % num_cor_pixel_utilizada)

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = int(input("Entre com numero de clusters (2 a 10): "))
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convertendo novamente em uint8, 
center = np.uint8(center)

# p ara cada valor em label.flatten(), estamos acessando posição correspondente no array center.
res = center[label.flatten()]
res2 = res.reshape((img.shape))
final = np.concatenate((img, res2), axis=1)
cv2.imshow('res2', final)
num = center.shape[0]
k = np.ones((K, 100, 100, 3), np.uint8)
for j in range(0, num):
    print("Centro do cluster %d:" % j)
    print(center[j])

# montando a barra de cores utilizadas na imagem resultante
for j in range(0, num):
    k[j, :, :, 0] = [center[j][0]]
    k[j, :, :, 1] = [center[j][1]]
    k[j, :, :, 2] = [center[j][2]]
'''
k2[:,:,0]=[center[1][0]]
k2[:,:,1]=[center[1][1]]
k2[:,:,2]=[center[1][2]]

k3[:,:,0]=[center[2][0]]
k3[:,:,1]=[center[2][1]]
k3[:,:,2]=[center[2][2]]
'''
concatenadas = []

# Itera sobre cada imagem em k e a adiciona à lista concatenadas
for imagem in k:
    concatenadas.append(imagem)

# Concatena todas as imagens na lista concatenadas ao longo do eixo horizontal (axis=1)
final = np.concatenate(concatenadas, axis=1)
# final=np.concatenate((k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]),axis=1)

cv2.imshow("cores dos clusters", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
