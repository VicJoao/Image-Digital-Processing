import cv2
import matplotlib.pyplot as plt
import numpy as np
'''

O comando inRange da OpenCV é usado para criar uma máscara que define uma faixa de valores dentro de um intervalo especificado em uma imagem.
Utilizada principalmente  em segmentação de objetos com base em sua cor.
Verifica cada pixel na imagem de entrada. Se o valor do pixel estiver dentro do intervalo especificado (entre limite_inferior e limite_superior),
o valor correspondente na máscara será definido como 255 (ou True, se estiver usando uma máscara binária).
Caso contrário, o valor do pixel na máscara será definido como 0 (ou False, se estiver usando uma máscara binária)

cv2.inRange(imagem, limite_inferior, limite_superior, máscara)
        imagem: A imagem de entrada na qual você deseja realizar a operação de inRange.
        limite_inferior: O limite inferior do intervalo de cor que você deseja detectar na imagem.
        limite_superior: O limite superior do intervalo de cor que você deseja detectar na imagem.
        máscara (opcional): Uma matriz de saída que define a região onde os valores de pixel estão dentro do intervalo especificado.
        Se não for fornecido, uma nova matriz será criada.
'''

lower_green = np.array([0, 220, 0])     ##[R value, G value, B value]
upper_green = np.array([60, 255, 60])

videoV=cv2.VideoCapture("thanos.mp4")
videoP=cv2.VideoCapture("Praieiro.mp4")

# Obter as propriedades do vídeo de entrada
frame_width = int(videoP.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(videoP.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(videoP.get(cv2.CAP_PROP_FPS))
total_frames = int(videoP.get(cv2.CAP_PROP_FRAME_COUNT))
# Configurar o codec de saída
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Criar o objeto VideoWriter para o vídeo de saída
video_writer = cv2.VideoWriter("Thanos na praia.mp4", fourcc, fps, (frame_width, frame_height))



contador=0
i=0
while True:
    ret,background_image = videoP.read()
    contador=contador+1
    if contador>400:
        ret,frameV=videoV.read()
        if ret:
            image_copy = np.copy(frameV)
            mask = cv2.inRange(image_copy, lower_green, upper_green)
            masked_image = np.copy(image_copy)
            #masked_image[mask != 0] = [0, 0, 0] define os pixels na imagem masked_image
            #que na náscara estão como não nulos para ter a cor preta. 
            masked_image[mask != 0] = [0, 0, 0]
            cv2.imshow("mascara",masked_image)
            #cv2.imshow("saida2", masked_image)
            background_image=cv2.resize(background_image,(image_copy.shape[1],image_copy.shape[0]))
            crop_background = background_image[0:image_copy.shape[0], 0:image_copy.shape[1]]
            # o comando crop_background[mask == 0] = [0, 0, 0] define os pixels na imagem crop_background que correspondem a regiões onde a máscara é nula
            #(ou seja, onde a imagem de fundo precisa ser ocultada) para ter a cor preta, removendo ou ocultando essas regiões da imagem de fundo. 
            crop_background[mask == 0] = [0, 0, 0]
            background_image = crop_background + masked_image

    cv2.imshow("saida", background_image)
    cv2.waitKey(1)
    video_writer.write(background_image)

    # Mostrar o progresso
    print("Processando quadro {}/{}".format(i, total_frames))
    i=i+1
    if cv2.waitKey(25)==27:
      break
videoV.release()
videoP.release()
video_writer.release()

#video2.release()
cv2.destroyAllWindows()

