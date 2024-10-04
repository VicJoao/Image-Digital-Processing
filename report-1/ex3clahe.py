import cv2

gray_img = cv2.imread('estatua.png',0)

equ_img = cv2.equalizeHist(gray_img)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
clahe_img = clahe.apply(gray_img)

cv2.imshow('Imagem original', gray_img)
cv2.imshow('Imagem equalizda', equ_img)
cv2.imshow('Imagem equalizda com CLAHE', clahe_img)

cv2.waitKey(0)
cv2.destroyAllWindows()