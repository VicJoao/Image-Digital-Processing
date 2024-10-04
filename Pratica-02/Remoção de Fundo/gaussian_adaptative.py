import cv2

# Carregar o vídeo
video_path = 'cars2.mp4'
cap = cv2.VideoCapture(video_path)


# Funções de limiarização adaptativa
def adaptive_mean_threshold(image, max_value=255, block_size=11, constant=2):
    return cv2.adaptiveThreshold(image, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)


def adaptive_gaussian_threshold(image, max_value=255, block_size=11, constant=2):
    return cv2.adaptiveThreshold(image, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size,
                                 constant)


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

    # Aplicar a limiarização adaptativa com média
    mean_binary = adaptive_mean_threshold(diff)

    # Aplicar a limiarização adaptativa com Gaussiana
    gaussian_binary = adaptive_gaussian_threshold(diff)

    # Exibir as imagens na tela
    cv2.imshow("Adaptive Mean Threshold", mean_binary)
    cv2.imshow("Adaptive Gaussian Threshold", gaussian_binary)

    # Atualizar o frame anterior
    gray_prev = gray

    # Aguardar 50 milissegundos entre cada frame
    cv2.waitKey(50)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
