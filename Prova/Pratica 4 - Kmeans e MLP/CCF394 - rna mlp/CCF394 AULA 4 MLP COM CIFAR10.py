import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.datasets import fetch_openml

# Carregar o conjunto de dados Fashion MNIST
fashion_mnist = fetch_openml('Fashion-MNIST')
print("Dados carregados")
# Dividir os dados em recursos (features) e rótulos (labels)
X = fashion_mnist.data
y = fashion_mnist.target.astype(int)
print("Tamanho do vetor X %d" %X.size)
# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("iniciando treinamento")
# Criar e treinar o classificador MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=150, random_state=42, verbose=1)
mlp.fit(X_train, y_train)
print("iniciando previsoes")
# Previsões no conjunto de teste
y_pred = mlp.predict(X_test)
'''
# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
'''
# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)
print('Acurácia {:.2f}'.format(100*accuracy))

from sklearn.preprocessing import label_binarize

# Convertendo os rótulos para o formato binário
y_bin = label_binarize(y_test, classes=np.unique(y))

# Calculando as probabilidades previstas para todas as classes
y_scores = mlp.predict_proba(X_test)

# Calculando a curva ROC para cada classe
#Para calcular a curva ROC em um problema de classificação multiclasse,
# utiliza-se  uma abordagem conhecida como "one-vs-all" ou "one-vs-rest".
#Nessa abordagem, você calcula uma curva ROC para cada classe em relação às demais.
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(np.unique(y))):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calculando a curva ROC média (macro)
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(np.unique(y)))]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(np.unique(y))):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= len(np.unique(y))
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plotando a curva ROC
plt.figure()
plt.plot(fpr["macro"], tpr["macro"], color='darkorange', lw=2, label='Curva ROC macro (area = %0.2f)' % roc_auc["macro"])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()



# Calcular a acurácia utilizando validação cruzada
cv_accuracy = cross_val_score(mlp, X, y, cv=5, scoring='accuracy').mean()
print("3")
# Previsões usando validação cruzada
y_cv_pred = cross_val_predict(mlp, X, y, cv=5)
print("4")
# Calcular a matriz de confusão usando validação cruzada
conf_matrix_cv = confusion_matrix(y, y_cv_pred)
print("5")
# Imprimir a acurácia e a matriz de confusão


print("Acurácia com validação cruzada:", cv_accuracy)
print("Matriz de Confusão:")
print(conf_matrix)
print("Matriz de Confusão com validação cruzada:")
print(conf_matrix_cv)

# Plotar a curva de treinamento
plt.figure()
plt.plot(mlp.loss_curve_, color='blue', label='Curva de Treinamento')
plt.xlabel('Número de Iterações')
plt.ylabel('Loss')
plt.title('Curva de Treinamento do MLP')
plt.legend(loc="upper right")
plt.show()

