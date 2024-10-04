import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#https://medium.com/@urapython.community/perceptron-com-python-uma-introdu%C3%A7%C3%A3o-f19aaf9e9b64

import numpy as np

# Dados de entrada e saída
X = np.array([[2, 1, 0], [1, 2, 1], [3, 2, 1], [2, 3, 0], [1, 1, 1], [3, 1, 1], [2, 2, 0], [1, 3, 1]], dtype=float)
y = np.array([20.6, 22.10, 35.53, 33, 16, 30.36, 26.8, 28.28], dtype=float)

# Normalizando os dados de entrada
X /= np.amax(X, axis=0)

# Inicializando os pesos aleatoriamente com valores entre -1 e 1
np.random.seed(1)
w = 2 * np.random.random((3, 1)) - 1

# Taxa de aprendizado
lr = 0.01
erros=[]
# Loop de treinamento

for i in range(1000):
    j = np.random.randint(0, len(X))
    output = np.dot(X[j],w)
    erro = y[j] - output
    w += lr * erro * X[j].reshape(3,1)
    erros.append(erro)



# Teste da rede com novos dados
X_test = np.array([[2, 2, 2], [1, 1, 1], [3, 3, 3]], dtype=float)
X_test /= np.amax(X_test, axis=0)
predicted_output = np.dot(X_test, w)

print(predicted_output)
plt.plot(range(1,len(erros)+1), erros, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Erros')
plt.show()
saidas=[]
for i in range(len(y)):
    saida= np.dot(X[i],w)
    saidas.append(saida)
print("-----------------------------------------------------------------------------------")
print("   saida  versus saida predita")
table = np.concatenate((y.reshape(-1,1), saidas), axis=1)
print(table)
print("-----------------------------------------------------------------------------------")
print("   saida  - saida predita  (erro na saida")
for i in range(len(y)):
    print(y[i]-saidas[i])

