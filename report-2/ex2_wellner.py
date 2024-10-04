import cv2
import numpy as np

def wellner_adaptive_threshold(img, window_size=15):
    height, width = img.shape
    binary_img = np.zeros_like(img, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            y1 = max(0, y - window_size // 2)
            y2 = min(height, y + window_size // 2 + 1)
            x1 = max(0, x - window_size // 2)
            x2 = min(width, x + window_size // 2 + 1)
            roi = img[y1:y2, x1:x2]
            mean = np.mean(roi)
            threshold = mean
            if img[y, x] > threshold:
                binary_img[y, x] = 255
    return binary_img

# Carrega o vídeo
video_path = 'cars2.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresholded_img_wellner = wellner_adaptive_threshold(gray)
    cv2.imshow('Wellner Video', thresholded_img_wellner)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
