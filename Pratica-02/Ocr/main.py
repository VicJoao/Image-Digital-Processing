import cv2
import numpy as np
import pytesseract
import re


# Função para ler a placa do carro com OCR
def ler_placa_sem_processamento(img, arquivo):
    result = pytesseract.image_to_string(img, config='--psm 6')
    cv2.imwrite(f"{arquivo}_ocr.png", img)
    return result.strip()


# Função para aplicar limiarização de Otsu
def aplicar_limiarizacao_otsu(img, arquivo):
    # Converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplica um filtro de mediana para remover ruídos
    gray = cv2.medianBlur(gray, 5)

    # Aplica a limiarização de Otsu
    _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"{arquivo}_limiarizacao.png", th_otsu)
    return th_otsu


# Função para segmentar a placa do carro
def segmentar_placa(img, arquivo):
    # Encontrar contornos
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar o maior contorno (placa do carro)
    max_contour = max(contours, key=cv2.contourArea)

    # Criar uma máscara para a placa do carro
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)

    # Aplicar a máscara na imagem original
    segmented_plate = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(f"{arquivo}_segmentada.png", segmented_plate)
    return segmented_plate


# Função para validar a placa do carro
def validar_placa(placa):
    # Expressão regular para verificar o padrão da placa
    pattern = r'^[A-Z]{3}\d[A-Z]\d{2}$'

    # Verifica se a placa corresponde ao padrão
    if re.match(pattern, placa):
        return True
    else:
        return False


# Leitura da imagem
imagem = cv2.imread('placas.png')
nome_arquivo = 'placas'

# Realiza a leitura da placa sem processamento
placa_sem_processamento = ler_placa_sem_processamento(imagem, nome_arquivo)
print("Placa do carro sem processamento:", placa_sem_processamento)

# Aplica limiarização de Otsu
imagem_limiarizada = aplicar_limiarizacao_otsu(imagem, nome_arquivo)

# Segmentação da placa
placa_segmentada = segmentar_placa(imagem_limiarizada, nome_arquivo)

# Realiza a leitura da placa com processamento
placa_com_processamento = ler_placa_sem_processamento(placa_segmentada, f"{nome_arquivo}_segmentada")
print("Placa do carro com processamento (antes da validação):", placa_com_processamento)

# Valida a placa
if validar_placa(placa_com_processamento):
    print("Placa do carro válida:", placa_com_processamento)
else:
    print("Placa do carro inválida ou não reconhecida.")
