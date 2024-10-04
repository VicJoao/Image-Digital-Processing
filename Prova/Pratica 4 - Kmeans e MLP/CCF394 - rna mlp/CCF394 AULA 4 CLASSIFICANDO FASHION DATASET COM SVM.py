import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from keras.datasets import fashion_mnist

# Carregando o conjunto de dados Fashion MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# Transformando as imagens em vetores unidimensionais
X_train = np.reshape(X_train, (60000, 784))
X_test = np.reshape(X_test, (10000, 784))

# Imprimindo o tamanho dos vetores de treinamento e teste
print("Tamanho do vetor de treinamento:", X_train.shape)
print("Tamanho do vetor de teste:", X_test.shape)

# Dividindo o conjunto de treinamento em conjunto de treinamento e conjunto de validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Definindo o classificador SVM com kernel linear
#clf = svm.SVC(kernel='linear',verbose=True)
# Definindo o classificador SVM com kernel linear
clf = svm.SVC(kernel='rbf')
print("treinando")
# Treinando o classificador SVM

costs = []
for i in range(100):
    clf.fit(X_train, y_train)
    costs.append(clf._get_coef()[0,0])

# Plota os valores da função custo durante o treinamento
plt.plot(costs)
plt.xlabel('Número de iterações')
plt.ylabel('Valor da função custo')
plt.show()



print("realizando a etapa de teste")
# Fazendo as previsões no conjunto de validação
y_pred = clf.predict(X_val)

# Calculando a acurácia e a matriz de confusão
accuracy = metrics.accuracy_score(y_val, y_pred)
confusion_matrix = metrics.confusion_matrix(y_val, y_pred)

# Exibindo a acurácia e a matriz de confusão
print("Acurácia:", accuracy)
print("Matriz de confusão:\n", confusion_matrix)

# Exibindo 10 imagens do conjunto de dados com as previsões
fig, axs = plt.subplots(2, 5)
fig.suptitle('Previsões do SVM no Fashion Dataset')

for i in range(2):
    for j in range(5):
        idx = np.random.randint(len(X_val))
        img = np.reshape(X_val[idx], (28, 28))
        axs[i,j].imshow(img, cmap='gray')
        axs[i,j].set_title("Previsto: %d\nReal: %d" % (y_pred[idx], y_val[idx]))
        axs[i,j].axis('off')

plt.show()
