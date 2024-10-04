import cv2
import numpy as np

"""
Neste código, primeiro carregamos uma imagem e definimos uma lista de valores gamma para correção gamma. Em 
seguida, iteramos sobre cada valor gamma, aplicamos a correção gamma na imagem e armazenamos as imagens corrigidas em 
uma lista. Por fim, concatenamos as imagens original e corrigidas lado a lado usando np.hstack() e exibimos a imagem 
resultante em uma janela OpenCV. O usuário pode pressionar qualquer tecla para fechar a janela.

O parâmetro gamma é uma medida que desempenha um papel importante na correção de gamma de imagens. Esse parâmetro 
controla a relação entre os valores dos pixels de entrada e de saída durante o processo de correção gamma, 
afetando diretamente o contraste e a luminosidade da imagem. Quando o valor de gamma é menor que 1, 
a imagem resultante tende a ser mais clara, suavizando as sombras e os realces. Por outro lado, um valor de gamma 
maior que 1 tende a escurecer a imagem, realçando os detalhes e aumentando o contraste. Assim, o ajuste do parâmetro 
gamma permite uma flexibilidade significativa na manipulação da aparência visual das imagens, adaptando-as às 
necessidades específicas de visualização e análise.
"""

# Carrega a imagem
img = cv2.imread('wiki.png')

# Define os valores dos parâmetros gamma
gamma_values = [1.5, 2]

# Lista para armazenar as imagens corrigidas
imagens_corrigidas = []

# Aplica a correção gama para cada valor de gamma
for gamma in gamma_values:
    # Aplica a correção gama na imagem
    img_corrigida = np.power(img/255.0, gamma)
    img_corrigida = np.uint8(img_corrigida*255)
    imagens_corrigidas.append(img_corrigida)

# Concatena as imagens original e corrigidas lado a lado
imagens_lado_a_lado = np.hstack([img] + imagens_corrigidas)

# Mostra as imagens original e corrigidas lado a lado
cv2.imshow('Imagem Original e Corrigida', imagens_lado_a_lado)
cv2.waitKey(0)

# Fecha a janela
cv2.destroyAllWindows()
