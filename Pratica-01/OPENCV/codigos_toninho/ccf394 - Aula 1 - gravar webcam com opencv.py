import numpy as np 
import cv2 
cap = cv2.VideoCapture(0)   # cap = cv2.VideoCapture("video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480)) 
contador=0
while(True): 
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    out.write(frame)
    if(contador % 10 ==0):
        cv2.imwrite("imagemRecortada"+str(contador)+".png", frame )
    cv2.imshow('Original', frame)
    contador+=1
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release() 
out.release()  
cv2.destroyAllWindows() 
