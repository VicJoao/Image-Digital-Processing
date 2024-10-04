import cv2
import numpy as np

def niblack_threshold(img, window_size=15, k=-0.2):
    mean = cv2.blur(img, (window_size, window_size))
    mean_square = cv2.blur(img**2, (window_size, window_size))
    std_dev = np.sqrt(mean_square - mean**2)
    threshold = mean + k * std_dev
    binary = img > threshold
    return binary.astype(np.uint8) * 255

# Carrega o vídeo
video_path = 'cars2.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresholded_img_niblack = niblack_threshold(gray)
    cv2.imshow('Niblack', thresholded_img_niblack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
