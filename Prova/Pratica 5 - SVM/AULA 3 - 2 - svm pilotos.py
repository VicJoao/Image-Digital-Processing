#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        title = 'Normalized confusion matrix' if normalize else 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix:")
    else:
        print('Confusion matrix, without normalization:')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

print("Script que lê o arquivo 'dados.csv' gerado pelo CCF394 roi picker de imagens")
print("E realiza a classificação dos pixels utilizando SVM")
print("Aceita somente 5 classes, determinadas durante a captura dos pixels")

df = pd.read_csv('dados.csv')
X = df.iloc[:, :3].values
y = df.iloc[:, 3].values

class_names = np.array(['Grupo1', 'Grupo2', 'Grupo3', 'Grupo4', 'Grupo5'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

classificador = svm.SVC(kernel='linear', C=0.01)
y_pred = classificador.fit(X_train, y_train).predict(X_test)

# Avaliando a precisão do modelo
precisao = accuracy_score(y_test, y_pred)
print(f'Precisão do modelo: {precisao * 100:.2f} %')

# Plotando a matriz de confusão
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plot_confusion_matrix(y_test, y_pred, classes=class_names, title='Confusion matrix, without normalization')

plt.subplot(1, 2, 2)
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()

print("Aguarde a classificação dos pixels...")

# Lendo a imagem
newImage = cv2.imread("pilotospequeno.png")

# Verificando se a imagem foi carregada corretamente
if newImage is None:
    print("Erro ao carregar a imagem!")
else:
    # Aplicando a classificação aos pixels da imagem
    rows, cols, _ = newImage.shape
    for y in range(rows):
        for x in range(cols):
            entrada = newImage[y, x].reshape(1, -1)
            saida = classificador.predict(entrada)
            if saida == 0:
                newImage[y, x] = (0, 0, 255)  # Vermelho
            elif saida == 1:
                newImage[y, x] = (255, 0, 0)  # Azul
            elif saida == 2:
                newImage[y, x] = (0, 255, 0)  # Verde
            elif saida == 3:
                newImage[y, x] = (255, 255, 255)  # Branco
            elif saida == 4:
                newImage[y, x] = (100, 100, 100)  # Cinza

    # Exibindo a imagem resultante
    cv2.imshow("Saída", newImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
