import cv2

# Carregar o vídeo
video_path = 'cars2.mp4'
cap = cv2.VideoCapture(video_path)

# Ler o primeiro frame
ret, frame_prev = cap.read()
gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
contador = 0
while True:
    # Ler o próximo frame
    ret, frame = cap.read()
    if not ret:
        break

    frame[:, :, 2] = cv2.equalizeHist(frame[:, :, 2])  # aplica equalização no canal V

    # Converter para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular a diferença entre os frames consecutivos
    diff = cv2.absdiff(gray, gray_prev)

    # Binarizar a imagem de diferença
    _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Aplicar operações de dilatação

    cv2.imshow("binary", binary)
    # Aplicar a máscara na imagem original
    masked_frame = cv2.bitwise_and(frame, frame, mask=binary)

    # Exibir a imagem na tela
    cv2.imshow("Masked Frame", masked_frame)
    cv2.waitKey(100)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Atualizar o frame anterior
    gray_prev = gray

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
