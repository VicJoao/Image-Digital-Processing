import cv2
import numpy as np

"""
O código apresentado realiza a transformação logarítmica em uma imagem em tons de cinza. Essa transformação é 
aplicada para aumentar o contraste em regiões de baixa intensidade e reduzir o contraste em regiões de alta 
intensidade. A técnica utiliza a função logarítmica para mapear os valores de pixel originais para novos valores, 
de modo que diferenças menores entre os valores originais resultem em diferenças mais perceptíveis na imagem de 
saída. A constante "c" é calculada com base no valor máximo dos pixels da imagem original para garantir que o 
intervalo de intensidade da imagem de saída seja entre 0 e 255. O resultado final é uma imagem com maior contraste e 
melhor visualização de detalhes, especialmente em áreas de baixa luminosidade.
"""

# Read an image
image = cv2.imread('menina.png', cv2.IMREAD_GRAYSCALE)

# Apply log transformation method
c = 255 / np.log(1 + np.max(image))
print(np.max(image))
print(c)
log_image = c * (np.log(image + 1))

# Specify the data type so that
# float value will be converted to int
log_image = np.array(log_image, dtype=np.uint8)

# Display both images
cv2.imshow("Entrada", image)

cv2.imshow("saida", log_image)
cv2.waitKey(0)
