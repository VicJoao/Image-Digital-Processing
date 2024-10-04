import cv2
import numpy as np

"""
O código implementa diferentes técnicas de limiarização em uma imagem em escala de cinza chamada "lena.jpg". 
Limiarização é o processo de converter uma imagem em escala de cinza em uma imagem binária, onde os pixels são 
classificados como brancos (valor 255) ou pretos (valor 0) com base em um limite (limiar) escolhido. O código aplica 
o método de Otsu, que automaticamente calcula o limite ótimo com base na variância dos níveis de cinza da imagem. 
Além disso, ele aplica outras técnicas de limiarização, como Niblack, Kapur, Kittler, Huang e Li, cada uma com suas 
próprias abordagens para determinar o limite ótimo. Os resultados são exibidos em diferentes janelas, permitindo a 
comparação visual entre as diferentes técnicas de limiarização.
"""


def otsu_threshold(img):
    # Compute histogram
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # Normalize histogram
    hist_norm = hist.astype(np.float32) / hist.sum()

    # Compute mean intensity
    np.arange(256).dot(hist_norm)

    # Initialize variables
    best_thresh = 0
    best_var = 0

    # Iterate over all possible thresholds
    for thresh in range(256):
        # Compute class probabilities
        prob1 = hist_norm[:thresh].sum()
        prob2 = 1 - prob1

        # Check if either class is empty
        if prob1 == 0 or prob2 == 0:
            continue

        # Compute class means
        mean1 = np.arange(thresh).dot(hist_norm[:thresh]) / prob1
        mean2 = np.arange(thresh, 256).dot(hist_norm[thresh:]) / prob2

        # Compute class variances
        var1 = np.power(np.arange(thresh) - mean1, 2).dot(hist_norm[:thresh]) / prob1
        var2 = np.power(np.arange(thresh, 256) - mean2, 2).dot(hist_norm[thresh:]) / prob2

        # Compute weighted within-class variance
        within_var = prob1 * var1 + prob2 * var2

        # Check if within-class variance is best so far
        if within_var > best_var:
            best_var = within_var
            best_thresh = thresh

    return best_thresh


# Carrega a imagem em escala de cinza
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Método de Otsu
otsu_thresh, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Limiarização de adaptação local (Niblack)
niblack_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Limiarização de entropia (Kapur)
kapur_thresh, kapur_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kapur_entropy = cv2.threshold(img, kapur_thresh, 255, cv2.THRESH_BINARY)[1]

# Limiarização de energia (Kittler)
kittler_thresh, kittler_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kittler_energy = cv2.threshold(img, kittler_thresh, 255, cv2.THRESH_BINARY)[1]

# Limiarização baseada em histograma (Huang)
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
hist_norm = hist.astype(np.float32) / img.size
cdf = hist_norm.cumsum()
bins_centers = (bins[:-1] + bins[1:]) / 2.
thresh_idx = np.argmax(cdf >= 0.5)
huang_thresh = bins_centers[thresh_idx]
huang_img = cv2.threshold(img, huang_thresh, 255, cv2.THRESH_BINARY)[1]

# Limiarização de variação mínima (Li)
li_thresh, li_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
li_var = np.var(np.concatenate([img[li_img == 0], img[li_img == 255]]))
li_thresh2 = (li_thresh + otsu_thresh) / 2 + li_var / (4 * (li_thresh - otsu_thresh))
li_img2 = cv2.threshold(img, li_thresh2, 255, cv2.THRESH_BINARY)[1]

# Mostra os resultados
cv2.imshow('Original', img)
cv2.imshow('Otsu', otsu_img)
cv2.imshow('Niblack', niblack_img)
cv2.imshow('Kapur', kapur_entropy)
cv2.imshow('Kittler', kittler_energy)
cv2.imshow('Huang', huang_img)
cv2.imshow('Li', li_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
