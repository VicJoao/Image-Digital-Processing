import cv2
import numpy as np


# Função de limiarização de Welnner
def apply_welnner_threshold(window_size, k):

    # Calcula a média local usando uma janela deslizante
    mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))

    # Aplica a fórmula de Wellner para calcular o limiar
    threshold = mean * (1 + k * (gray - mean) / mean)

    # Aplica o limiar na imagem
    thresholded_img = np.zeros_like(gray)
    thresholded_img[gray >= threshold] = 255

    return thresholded_img


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

    frame[:, :, 2] = cv2.equalizeHist(frame[:, :, 2])  # aplica equalização no canal V

    # Converter para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular a diferença entre os frames consecutivos
    diff = cv2.absdiff(gray, gray_prev)

    # Aplicar a limiarização de Niblack
    wellner_binary = apply_welnner_threshold(window_size=50, k=-0.9)

    # Exibir as imagens na tela
    cv2.imshow("Wellner", wellner_binary)

    # Atualizar o frame anterior
    gray_prev = gray

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
