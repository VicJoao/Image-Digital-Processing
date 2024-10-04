import cv2
import numpy as np

'''
Após a equalização do histograma de uma imagem, a função de distribuição acumulada
(CDF - Cumulative Distribution Function) torna-se linear devido ao processo de equalização. Vamos entender o motivo:
Equalização do histograma: Durante a equalização do histograma, a ideia é distribuir as intensidades de pixel da imagem de forma mais uniforme,
o que significa que os pixels de intensidades baixas e altas serão estendidos para ocupar uma faixa maior de intensidades.
Isso é feito alterando os valores de intensidade dos pixels para tornar a distribuição de intensidades mais uniforme.

CDF (Função de Distribuição Acumulada): A CDF é calculada a partir do histograma da imagem original.
Ela mostra a probabilidade cumulativa de encontrar um pixel com uma determinada intensidade na imagem.
Ou seja, ela representa a probabilidade de encontrar um pixel com intensidade menor ou igual a um determinado valor.

Linearidade após a equalização: Após a equalização do histograma, a CDF torna-se linear. Isso ocorre porque,
após a equalização, a distribuição de intensidades de pixel é uniformizada. Portanto,
a probabilidade cumulativa de encontrar um pixel com uma intensidade específica torna-se linear.
'''

from matplotlib import pyplot as plt
img = cv2.imread('wiki.png',0)
equ = cv2.equalizeHist(img)  #retorna a imagem equalizada
#histograma e cdf  da imagem original
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.subplot(221)
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.title('Histograma imagem original'), plt.xticks([]), plt.yticks([])
#histograma e cdf da imagem equalizada
hist,bins = np.histogram(equ.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.subplot(222)
plt.plot(cdf_normalized, color = 'b')
plt.hist(equ.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.title('Histograma imagem equalizada '), plt.xticks([]), plt.yticks([])
plt.subplot(223)
# o formato opencv é BGR, e o plot usa RGB, entao usa COLOR_BGR2RGB
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(224)
plt.imshow(cv2.cvtColor(equ, cv2.COLOR_BGR2RGB))
plt.show()


