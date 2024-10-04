import numpy as np
import matplotlib.pyplot as plt
import gc

# Definir os dados de entrada e saída
X = np.array([[2, 1, 0], [1, 2, 1], [3, 2, 1], [2, 3, 0], [1, 1, 1], [3, 1, 1], [2, 2, 0], [1, 3, 1]], dtype=float)
y = np.array([20.6, 22.10, 35.53, 33, 16, 30.36, 26.8, 28.28], dtype=float)

# Inicializar os pesos aleatoriamente
np.random.seed(1)
w1 = 2 * np.random.random((3, 3)) - 1
w2 = 2 * np.random.random((3, 1)) - 1

# Definir a função de ativação linear
def linear_activation(x):
    return x

# Definir a taxa de aprendizagem e o número de iterações
learning_rate = 0.01
num_iterations = 10

# Armazenar os erros durante a etapa de treinamento
errors = []

# Loop de treinamento
for i in range(num_iterations):

    # Propagação direta
    h1 = linear_activation(np.dot(X, w1))
    output = linear_activation(np.dot(h1, w2))

    # Calcular o erro
    error = y - output.flatten()
    errors.append(np.mean(np.abs(error)))  # armazenar o erro absoluto médio

    # Retropropagação
    output_delta = error * learning_rate
    h1_error = output_delta.reshape(-1, 1).dot(w2.T)
    h1_delta = h1_error * learning_rate

    # Atualizar os pesos
    w2 += h1.T.dot(output_delta.reshape(-1, 1))
    w1 += X.T.dot(h1_delta)

# Plotar os erros durante a etapa de treinamento
plt.plot(errors)
plt.title("Erro durante o treinamento")
plt.xlabel("Iteração")
plt.ylabel("Erro absoluto médio")
plt.show()

# Exibir uma tabela comparando a saída da rede com o vetor y
output_final = linear_activation(np.dot(linear_activation(np.dot(X, w1)), w2))
diferenca_saida = (y.reshape(-1,1) - output_final)
table = np.concatenate((y.reshape(-1,1), output_final), axis=1)
table = np.concatenate((table, diferenca_saida), axis=1)

print("Tabela de comparação entre y e a saída da rede:")
print(table)
gc.collect()
