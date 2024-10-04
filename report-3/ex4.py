import numpy as np
import cv2
import matplotlib.pyplot as plt

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global pixels
        global samples
        (B,G,R) = img[y, x]
        p = np.array([[B,G,R,classes]])
        pixels = np.concatenate((pixels,p))
        print("Amostra {}: {},{},{}".format(samples, B, G, R))
        samples = samples + 1

img = cv2.imread("halteres.jpg")
cv2.namedWindow('halteres')
cv2.setMouseCallback("halteres", on_mouse)
file = "dados_ex4.csv"
open(file, 'w').close()

for i in range(7):
    samples = 0
    classes = i
    pixels = np.zeros((1,4),dtype=np.int8)
    print("Amostras classe {}:".format(i))
    while True:
        cv2.imshow("halteres", img)
        cv2.waitKey(1)
        if samples >= 30:
            with open(file, 'a') as f:
                break
    print("fim da amostra {}".format(i))
    pixels = pixels[1:]
    avg = np.mean(pixels,axis=0)
    avgB = avg[0]
    avgG = avg[1]
    avgR = avg[2]
    rows, cols, rgb = img.shape
    csv_rows = (["{},{},{},{}\n".format(i, j, k,l) for i, j, k,l in pixels])
    csv_text = "".join(csv_rows)
    with open(file, 'a') as f:
        f.write(csv_text)

cv2.setMouseCallback("halteres", lambda *args : None)
f.close()

print("fim das amostras")

cv2.waitKey(0)
cv2.destroyAllWindows()