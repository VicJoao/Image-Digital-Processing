import cv2
import numpy as np

gray_img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Imagem cinza', gray_img)
img = cv2.imread('lena.jpg')
cv2.imshow('Imagem colorida', img)

gray_hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256], accumulate=False)
gray_hist_img = np.zeros((400, 512), dtype=np.uint8)
cv2.normalize(gray_hist, gray_hist, alpha=0, beta=400, norm_type=cv2.NORM_MINMAX)
for i in range(1, 256):
      cv2.line(gray_hist_img, (2 * (i - 1), 400 - int(gray_hist[i - 1])),
              (2 * (i), 400 - int(gray_hist[i])),
              (255, 0, 0), thickness=2)

cv2.imshow('Histograma cinza', gray_hist_img)

bgr = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
bgr_planes = cv2.split(img)
bgr_hist = []
hist_img = np.zeros((400, 512, 3), dtype=np.uint8)

for i in range(3):
        bgr_hist.append(cv2.calcHist(bgr_planes, [i], None, [256], [0, 256], accumulate=False))
        cv2.normalize(bgr_hist[i], bgr_hist[i], alpha=0, beta=400, norm_type=cv2.NORM_MINMAX)

        for j in range(1, 256):
                cv2.line(hist_img, (2 * (j - 1), 400 - int(bgr_hist[i][j - 1])),
                        (2 * (j + 1), 400 - int(bgr_hist[i][j])),
                        bgr[i], thickness=2)

cv2.imshow('Histograma colorido', hist_img)

while True:
    k = cv2.waitKey(0) & 0xFF     
    if k == 27: break             # ESC key to exit 
cv2.destroyAllWindows()