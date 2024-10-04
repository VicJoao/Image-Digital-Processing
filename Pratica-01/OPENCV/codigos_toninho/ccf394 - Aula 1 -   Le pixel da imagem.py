import cv2
import argparse
import glob
import numpy as np

# função de retorno de chamada do mouse
def mostrarValorPixel(event, x, y, flags, param):
    global img, resultadoCombinado, espacoCor

    if event == cv2.EVENT_MOUSEMOVE:
        # obter o valor do pixel a partir da posição do mouse em (x, y)
        bgr = img[y, x]

        # Converter o pixel BGR em outros formatos de cor

        # Criar um espaço reservado vazio para exibir os valores
        espacoCor = np.zeros((img.shape[0], 400, 3), dtype=np.uint8)

        # preencher o espaço reservado com os valores dos espaços de cor
        cv2.putText(espacoCor, "BGR {}".format(bgr), (20, 70), cv2.FONT_HERSHEY_COMPLEX, .9, (255, 255, 255), 1, cv2.LINE_AA)

        # Combinar os dois resultados para mostrar lado a lado em uma única imagem
        resultadoCombinado = np.hstack([img, espacoCor])

        cv2.imshow('PRESSIONE P para Anterior, N para Próxima Imagem', resultadoCombinado)


if __name__ == '__main__':

    # carregar a imagem e configurar a função de retorno de chamada do mouse
    global img
    arquivos = glob.glob('lena.png')
    arquivos.sort()
    img = cv2.imread(arquivos[0])
    cv2.imshow('PRESSIONE P para Anterior, N para Próxima Imagem', img)

    # Criar uma janela vazia

    # Criar uma função de retorno de chamada para qualquer evento do mouse
    cv2.setMouseCallback('PRESSIONE P para Anterior, N para Próxima Imagem', mostrarValorPixel)
    i = 0
    while(1):
        k = cv2.waitKey(1) & 0xFF
        # verificar a próxima imagem na pasta
        if k == ord('n'):
            i += 1
            img = cv2.imread(arquivos[i % len(arquivos)])
            img = cv2.resize(img, (400, 400))
            cv2.imshow('PRESSIONE P para Anterior, N para Próxima Imagem', img)

        # verificar a imagem anterior na pasta
        elif k == ord('p'):
            i -= 1
            img = cv2.imread(arquivos[i % len(arquivos)])
            img = cv2.resize(img, (400, 400))
            cv2.imshow('PRESSIONE P para Anterior, N para Próxima Imagem', img)

        elif k == 27:
            cv2.destroyAllWindows()
            break

#%%
