import cv2
from matplotlib import pyplot as plt

"""
O código dado realiza a captura de vídeo da webcam e exibe tanto o vídeo em cores quanto os histogramas das bandas 
RGB e da imagem em escala de cinza. Vamos melhorar o código dividindo-o em funções e simplificando algumas partes.
"""


def capture_and_show_histograms():
    # Configurações da câmera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Captura o frame da câmera
        ret, frame = cap.read()

        # Exibe o vídeo da webcam e os histogramas
        show_video_and_histograms(frame)

        # Sai do loop ao pressionar a tecla 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # Libera a câmera e fecha a janela
    cap.release()
    cv2.destroyAllWindows()


def show_video_and_histograms(frame):
    # Converte a imagem para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcula o histograma da imagem monocromática
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Separa as bandas de cores RGB
    b, g, r = cv2.split(frame)

    # Calcula os histogramas das bandas RGB
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

    # Exibe o vídeo da webcam e os histogramas
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Frame')

    plt.subplot(2, 3, 2)
    plt.plot(hist_gray, color='gray')
    plt.title('Histograma Escala de Cinza')

    plt.subplot(2, 3, 4)
    plt.plot(hist_r, color='r')
    plt.title('Histograma Vermelho')

    plt.subplot(2, 3, 5)
    plt.plot(hist_g, color='g')
    plt.title('Histograma Verde')

    plt.subplot(2, 3, 6)
    plt.plot(hist_b, color='b')
    plt.title('Histograma Azul')

    plt.tight_layout()
    plt.show()


# Função principal para iniciar a captura e exibir os histogramas
if __name__ == "__main__":
    capture_and_show_histograms()
