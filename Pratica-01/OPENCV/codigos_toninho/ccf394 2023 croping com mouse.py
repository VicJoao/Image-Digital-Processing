import cv2


# Função para cortar a imagem com base nos pontos de clique do usuário
def crop_image(event, x, y, flags, param):
    global point1, point2, cropping
    # Evento de clique do mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        if not cropping:
            point1 = (x, y)
            cropping = True
        else:
            point2 = (x, y)
            cropping = False
            # Corta a imagem com base nos pontos de clique
            cropped_image = image[point1[1]:point2[1], point1[0]:point2[0]]
            # Exibe a imagem cortada
            cv2.imshow("Cropped Image", cropped_image)
            cv2.imwrite("imagemRecortada.png", cropped_image)


# Leitura da imagem de entrada
image = cv2.imread("imagem.jpg")
# Inicialização das variáveis de controle
point1 = None
point2 = None
cropping = False
# Exibe a imagem original
cv2.imshow("Original Image", image)
# Define a função de clique do mouse
cv2.setMouseCallback("Original Image", crop_image)
# Loop principal do programa
while True:
    key = cv2.waitKey(1)
    # Encerra o programa ao pressionar a tecla "q"
    if key == ord("q"):
        break

# Libera os recursos utilizados pelo OpenCV
cv2.destroyAllWindows()

# %%
