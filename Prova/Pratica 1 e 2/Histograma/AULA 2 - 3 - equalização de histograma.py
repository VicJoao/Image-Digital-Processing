import cv2

# Carrega a imagem em tons de cinza
img = cv2.imread('wiki.png', 0)

# Equaliza a imagem
equ = cv2.equalizeHist(img)  # Retorna a imagem equalizada

# Mostra a imagem original e a imagem equalizada
cv2.imshow("Imagem Original", img)
cv2.imshow("Imagem Equalizada", equ)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
A equalização de histograma é um método de processamento de imagem que redistribui as intensidades dos pixels de 
forma que o histograma resultante seja mais uniforme. Isso geralmente é feito para melhorar o contraste da imagem e 
realçar detalhes. A equalização de histograma é particularmente útil em imagens com uma distribuição desigual de 
intensidades, como imagens que são muito escuras ou muito claras.

O código carrega uma imagem em tons de cinza ('wiki.png') e em seguida aplica a equalização de histograma usando a 
função cv2.equalizeHist(). A imagem original e a imagem equalizada são exibidas em janelas separadas usando 
cv2.imshow(). Após pressionar qualquer tecla, as janelas são fechadas usando cv2.destroyAllWindows().
"""