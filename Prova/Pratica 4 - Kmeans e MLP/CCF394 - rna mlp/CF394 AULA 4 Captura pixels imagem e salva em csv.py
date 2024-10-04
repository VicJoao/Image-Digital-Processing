import numpy as np
import numpy.linalg as la
import cv2
import math
import matplotlib.pyplot as plt
import time
# pergunta o numerode classes e coleta amostras de pixels para um arquivo csv
# digite 'c' quando terminar uma classe e quiser passar para coletar dados da proxima
classe=0
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #amostrando um quadrado de pixels
        global pixels
        (B,G,R)=img[y,x]
        print ("col: %d, row: %d   - R= %d, G=%d, B=%d" % (x, y, R,G,B))
        p=np.array([[B,G,R,classe]])
        print(x)
        pixels=np.concatenate((pixels,p))
        (B,G,R)=img[y,x+1]
        p=np.array([[B,G,R,classe]])
        pixels=np.concatenate((pixels,p))
        (B,G,R)=img[y+1,x]
        p=np.array([[B,G,R,classe]])
        pixels=np.concatenate((pixels,p))
         
        (B,G,R)=img[y+1,x+1]
        p=np.array([[B,G,R,classe]])
        pixels=np.concatenate((pixels,p))
        
        
arquivo="dados.csv"
   
numClasses=int(input("entre com o numero de classes para coletas pixels: "))
for j in range(0, numClasses):
    print("Coletando para o agrupamento %d " % j)
    pixels =np.zeros((1,4),dtype=np.int8)
    print(pixels)
    img=cv2.imread("pilotosPEQUENO.png")
    cv2.namedWindow('frame1')
    cv2.setMouseCallback("frame1", on_mouse)
    CONTADOR=0
    while True:
        # display the image and wait for a keypress
        cv2.imshow("frame1", img)
        key = cv2.waitKey(1) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            with open(arquivo, 'a') as f:
                f.write("\n")
            break
    classe=classe+1
    pixels = pixels[1:] #remove 000 inicial
    media=np.mean(pixels,axis=0)
    mediaB=media[0]
    mediaG=media[1]
    mediaR=media[2]
    print("Media R=%d, media G= %d, media B=%d \n " % (mediaR, mediaG, mediaB))
    rows,cols,cor = img.shape
    csv_rows = (["{},{},{},{}".format(i, j, k,l) for i, j, k,l in pixels])
    linha=None
    csv_text = "\n".join(csv_rows)
    with open(arquivo, 'a') as f:
        f.write(csv_text)
f.close()
cv2.destroyAllWindows()



    
    

