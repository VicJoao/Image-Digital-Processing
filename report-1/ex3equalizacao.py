import cv2
import numpy as np
from matplotlib import pyplot as plt
img_original = cv2.imread('wiki.png', 0)
img_equ = cv2.equalizeHist(img_original)

hist,bins = np.histogram(img_original.flatten(),256,[0,256])
plt.subplot(221)
plt.hist(img_original.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.xticks([0, 63, 127, 191, 255])
plt.title('Histograma da imagem original')

hist,bins = np.histogram(img_equ.flatten(),256,[0,256])
plt.subplot(222)
plt.hist(img_equ.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.xticks([0, 63, 127, 191, 255])
plt.title('Histograma da imagem equalizada')

plt.subplot(223)
plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)

plt.subplot(224)
plt.imshow(cv2.cvtColor(img_equ, cv2.COLOR_BGR2RGB))
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
plt.show()


