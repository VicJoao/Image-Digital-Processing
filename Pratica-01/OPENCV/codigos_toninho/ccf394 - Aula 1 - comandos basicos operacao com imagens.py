import cv2 as cv
import numpy as np
# Carregando a imagem do disco
img = cv.imread('../imagens/imagem.jpg')
rotated_image = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
cv.imshow("Rotacioanando",rotated_image)
cv.waitKey(0)


h,w,c=img.shape
print('Imagem colorida')
print(f"Altura {h}, Largura {w} , Canais {c}")
# Convertendo para escala de cinza
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
h,w=gray.shape
print('Imagem monocromatica')
print(f"Altura {h}, Largura {w} ")
# Invertendo horizontalmente
h_flip = cv.flip(img, 1)
# Invertendo verticalmente
v_flip = cv.flip(img, 0)
# Criando uma pilha vertical de imagens
stacked_img = np.vstack((img, h_flip, v_flip))
# Mostrando as imagens simultaneamente
cv.imshow('Imagens', stacked_img)
cv.waitKey(0)
cv.imshow('Imagens', gray)
cv.waitKey(0)
cv.destroyAllWindows()
b,g,r=cv.split(img)
stacked_img = np.vstack((b, g, r))
cv.imshow('Imagens', stacked_img)
cv.waitKey(0)
a=img[76:125,53:242]
cv.imshow("a",a)
cv.waitKey(0)
#trocando bandas
nova = cv.merge([r,b,g])
cv.imshow("Bandas Trocadas",nova)
cv.waitKey(0)
cv.destroyAllWindows()





