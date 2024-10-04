#!/usr/bin/env python
# coding: utf-8

# https://scikit-learn.org/stable/modules/svm.html

# neste script, o arquivo pilotospequeno.csv contem amostras dos pixels das classes previamente selecionadas.
# bgr são as cores e classe é a classe que o operador selecionou quando coletou os pixels
# b	g	r	classe
# 0	3	255	0


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import accuracy_score
import cv2


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
   
    if not title:
        if normalize:
            title = 'Matriz de Confusão normalizada'
        else:
            title = 'Matriz de Confusão NÃO normalizada'

    # calcula  a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusão normalizada")
    else:
        print('Matriz de Confusão NÃO normalizada')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ...criando os labels
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Entrada',
           xlabel='Saida')

    # rotacionando os labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Percorrendo sobre os dados e criando as anotações de texto
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),ha="center", va="center",color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[4]:
print("Script que le o arquivo dados.csv gerado pelo  CCF394 roi picker de imagens")
print("E realiza a classificação dos pixels utilizando SVM")
print("aceita somente 5 classes, determinadas durante a captura dos pixels")

df = pd.read_csv('pilotospequeno.csv')
df.head(2)


#selecionado os dados
X=df.iloc[:,[0,1,2]].values
Y=df.iloc[:,[3]].values


#criando o nome das 5 classes de saida
class_names = np.array(['Grupo1','Grupo2','Grupo3','Grupo4','Grupo5'])

#dividindo o pacote de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)



#criando o modelo do classificador e obtendo as saidas para as entradas de teste
classificador = svm.SVC(kernel='linear', C=0.01)
y_pred = classificador.fit(X_train, y_train).predict(X_test)


#convertendo para array, para aplicar na matriz de confusão
y_test=np.array(y_test)
y_pred=np.array(y_pred)


np.set_printoptions(precision=2)
#plot_decision_regions(X, Y, classifier=classificador)
#plt.legend(loc='upper left')
#plt.tight_layout()
#plt.show()


# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, title='Matriz de Confusão')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names , normalize=True,
                      title='Normalized confusion matrix')



precisao=100*accuracy_score(y_test,y_pred)
print(f'Precisão do modelo: {precisao:.3f} %')
print(" Aguarde a classificação dos pixels...")

newImage = cv2.imread("pilotospequeno.png")
rows,cols,cor = newImage.shape
for y in range(0,rows):
    for x in range(0,cols):
        entrada=newImage[y,x]
        Saida=classificador.predict(entrada.reshape(1,-1))
        if Saida==0:
            newImage[y,x]=(0,0,255)
        if Saida==1:
            newImage[y,x]=(255,0,0)
        if Saida==2:
            newImage[y,x]=(0,255,0)
        if Saida==3:
            newImage[y,x]=(255,255,255)
        if Saida==4:
            newImage[y,x]=(100,100,100)
cv2.imshow("saida",newImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
        
        





