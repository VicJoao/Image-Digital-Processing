import numpy as np
import cv2


# Função de callback para eventos de mouse, que captura os píxeis selecionados pelo usuário
def on_mouse(event, x, y, flags, param):
    global pixels
    if event == cv2.EVENT_LBUTTONDOWN:
        matrix_pixels = []

        print(f"Coordenadas do pixel: ({x}, {y})")
        # Adiciona os píxeis de uma janela 5x5 ao redor do píxel selecionado
        for i in range(-2, 3):
            for j in range(-2, 3):
                blue, green, red = img[y + j, x + i]
                matrix_pixels.append([blue, green, red])
        matrix_pixels = np.array(matrix_pixels)
        pixels = np.append(pixels, matrix_pixels, axis=0)


# Função para processar a imagem
def process_image(imagem, media_b, media_g, media_r, tol, tamanho_janela):
    rows, cols, _ = imagem.shape  # Obtém as dimensões da imagem

    # Itera sobre todos os píxeis da imagem
    for i in range(rows):
        for j in range(cols):
            count = 0  # Inicializa o contador de píxeis vizinhos dentro da tolerância

            # Itera sobre a janela de vizinhança ao redor do píxel atual
            for m in range(max(0, i - tamanho_janela // 2), min(rows, i + tamanho_janela // 2 + 1)):
                for n in range(max(0, j - tamanho_janela // 2), min(cols, j + tamanho_janela // 2 + 1)):
                    b_neighbor, g_neighbor, r_neighbor = imagem[m, n]  # Obtém a cor do píxel vizinho

                    # Verifica se a cor do pixel vizinho está dentro da tolerância especificada
                    if ((media_b - tol) < b_neighbor < (media_b + tol)) and \
                            ((media_g - tol) < g_neighbor < (media_g + tol)) and \
                            ((media_r - tol) < r_neighbor < (media_r + tol)):
                        count += 1  # Incrementa o contador se a cor do pixel vizinho estiver dentro da tolerância

            # Se a maioria dos píxeis vizinhos estiver dentro da tolerância, atualiza o píxel atual para a cor média
            if count >= (tamanho_janela * tamanho_janela) // 2:
                imagem[i, j] = (media_b, media_g, media_r)

    return imagem


# Carrega a imagem
img = cv2.imread("halteres.jpg")
print("Imagem carregada")

# Cria uma janela para exibir a imagem
cv2.namedWindow('frame1')

# Cria uma matriz para armazenar os píxeis selecionados pelo usuário
pixels = np.zeros((1, 3), dtype=np.uint8)

# Define uma matriz para armazenar as médias de cada cor
medias_cores = np.zeros((3, 6), dtype=np.uint8)  # 3 linhas para B, G e R; 6 colunas para as 6 cores

# Define a função de callback para eventos de mouse
cv2.setMouseCallback("frame1", on_mouse)

conjunto_cor = ["Azul", "Preto", "Amarelo", "Vermelho", "Cinza", "Verde"]

# Loop para capturar as 6 cores
for cor in range(6):
    print(f"Selecione a cor: {conjunto_cor[cor]}, pressione q para próxima cor.")
    while True:
        cv2.imshow("frame1", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Remove a primeira linha da matriz de píxeis, inicializada com zeros
    pixels = np.delete(pixels, 0, axis=0)

    # Calcula a média dos valores BGR dos pixels selecionados
    media = np.mean(pixels, axis=0)
    print("Média: ", media)
    mediaB = media[0]
    mediaG = media[1]
    mediaR = media[2]

    # Armazena a média na matriz de médias de cores
    medias_cores[:, cor] = [mediaB, mediaG, mediaR]
    print("Média da cor: ", mediaB, mediaG, mediaR)
    print(medias_cores)

    # Esvazia a matriz de pixels
    pixels = np.zeros((1, 3), dtype=np.uint8)

# Define uma margem de tolerância para os valores de cor
tolerancia = 60

# Define o tamanho da janela ao redor de cada pixel para verificar os vizinhos
window_size = 5

# Processa a imagem para cada uma das 6 cores

# Processando para Azul:
print("Processando para Azul")
azul = process_image(img.copy(), medias_cores[0, 0], medias_cores[1, 0], medias_cores[2, 0], tolerancia,
                     window_size)
cv2.imshow("Azul", azul)
cv2.waitKey(0)

# Processando para Preto:
print("Processando para Preto")
preto = process_image(azul.copy(), medias_cores[0, 1], medias_cores[1, 1], medias_cores[2, 1], tolerancia,
                      window_size)
cv2.imshow("Preto", preto)
cv2.waitKey(0)

# Processando para Amarelo:
print("Processando para Amarelo")
amarelo = process_image(preto.copy(), medias_cores[0, 2], medias_cores[1, 2], medias_cores[2, 2], tolerancia,
                        window_size)
cv2.imshow("Amarelo", amarelo)
cv2.waitKey(0)

# Processando para Vermelho:
print("Processando para Vermelho")
vermelho = process_image(amarelo.copy(), medias_cores[0, 3], medias_cores[1, 3], medias_cores[2, 3], tolerancia,
                         window_size)
cv2.imshow("Vermelho", vermelho)
cv2.waitKey(0)

# Processando para Cinza:
print("Processando para Cinza")
cinza = process_image(vermelho.copy(), medias_cores[0, 4], medias_cores[1, 4], medias_cores[2, 4], tolerancia,
                      window_size)
cv2.imshow("Cinza", cinza)
cv2.waitKey(0)

# Processando para Verde:
print("Processando para Verde")
verde = process_image(cinza.copy(), medias_cores[0, 5], medias_cores[1, 5], medias_cores[2, 5], tolerancia,
                      window_size)
cv2.imshow("Verde", verde)
cv2.waitKey(0)

for i in range(6):
    d = input("Digite seu nome:")
    print(d)


#cv2.destroyAllWindows()
