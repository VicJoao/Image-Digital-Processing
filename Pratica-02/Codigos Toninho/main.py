import cv2
import numpy as np


# Função para o limiar de Wellner adaptativo
def wellner_adaptive_threshold(img, window_size=15, k=0.02):
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
            std_dev = np.std(roi)
            threshold = mean + k * (std_dev - mean)
            if img[y, x] > threshold:
                binary_img[y, x] = 255
    return binary_img


# Carregar o vídeo
video_path = 'cars2.mp4'
cap = cv2.VideoCapture(video_path)

# Ler o primeiro frame
ret, frame_prev = cap.read()
gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)

while True:
    # Ler o próximo frame
    ret, frame = cap.read()
    if not ret:
        break

    # Converter para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular a diferença entre os frames consecutivos
    diff = cv2.absdiff(gray, gray_prev)

    # Binarizar a imagem de diferença usando Wellner adaptativo
    binary_wellner = wellner_adaptive_threshold(diff)

    # Aplicar operações de dilatação
    kernel = np.ones((5, 5), np.uint8)
    binary_wellner = cv2.dilate(binary_wellner, kernel, iterations=2)

    # Aplicar a máscara na imagem original
    masked_frame = cv2.bitwise_and(frame, frame, mask=binary_wellner)

    # Exibir a imagem na tela
    cv2.imshow("Masked Frame (Welnner)", masked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Atualizar o frame anterior
    gray_prev = gray

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
