import cv2

# Carregar o vídeo
video_path = 'cars2.mp4'
cap = cv2.VideoCapture(video_path)

# Ler o primeiro frame
ret, frame_prev = cap.read()
gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame[:, :, 2] = cv2.equalizeHist(frame[:, :, 2])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray, gray_prev)
    _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    cv2.imshow("binary", binary)
    masked_frame = cv2.bitwise_and(frame, frame, mask=binary)
    cv2.imshow("Original", masked_frame)

    cv2.waitKey(100)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    gray_prev = gray

cap.release()
cv2.destroyAllWindows()