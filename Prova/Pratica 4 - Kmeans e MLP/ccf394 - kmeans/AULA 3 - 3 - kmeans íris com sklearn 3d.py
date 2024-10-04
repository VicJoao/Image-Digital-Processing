# Importar bibliotecas necessárias
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

"""
Este script realiza o agrupamento dos dados do conjunto de dados Iris usando o algoritmo KMeans. Inicialmente, 
são carregados os dados do conjunto de dados Iris. Em seguida, é criado um modelo KMeans com 3 clusters, 
correspondendo ao número de espécies de íris no conjunto de dados. O modelo KMeans é treinado e usado para prever os 
rótulos dos clusters para cada amostra de dados. Um passo adicional mapeia os rótulos previstos para os rótulos reais 
das espécies de íris, permitindo a comparação com os rótulos verdadeiros. A acurácia do modelo KMeans é calculada em 
relação aos rótulos verdadeiros. Em seguida, os dados são plotados em 3D antes e depois do agrupamento, 
para visualizar a separação dos clusters. Os centroides resultantes do agrupamento também são plotados para indicar a 
posição de cada cluster no espaço de características.
"""

# Carregar conjunto de dados Iris
iris = load_iris()

# Definir variáveis de entrada e saída
X = iris.data
y = iris.target

# Criar modelo KMeans com 3 clusters (correspondente ao número de espécies de iris)
kmeans = KMeans(n_clusters=3, random_state=0)

# Executar modelo KMeans
y_kmeans = kmeans.fit_predict(X)

y_pred = kmeans.labels_

# Passo adicional: Mapear os rótulos previstos para os rótulos reais
label_mapping = {}
for i in range(3):
    label = np.argmax(np.bincount(y[y_pred == i]))
    label_mapping[i] = label

y_pred_mapped = np.array([label_mapping[label] for label in y_pred])

# Passo 5: Calcular a acurácia
accuracy = 100 * accuracy_score(y, y_pred_mapped)

# Passo 6: Imprimir a acurácia
print("Acurácia do KMeans no conjunto de dados Iris:", accuracy)

# Calcular acurácia da separação
accuracy = accuracy_score(y, y_kmeans)

# Plotar dados antes do agrupamento em 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
ax.set_xlabel('Comprimento da Sépala (cm)')
ax.set_ylabel('Largura da Sépala (cm)')
ax.set_zlabel('Comprimento da Pétala (cm)')
ax.set_title('Dados Iris Antes do Agrupamento')
plt.show()

# Plotar dados depois do agrupamento em 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_kmeans)
ax.set_xlabel('Comprimento da Sépala (cm)')
ax.set_ylabel('Largura da Sépala (cm)')
ax.set_zlabel('Comprimento da Pétala (cm)')
ax.set_title('Dados Iris Depois do Agrupamento')
plt.show()

# Plotar centroides resultantes em 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_kmeans)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=100, c='red')
ax.set_xlabel('Comprimento da Sépala (cm)')
ax.set_ylabel('Largura da Sépala (cm)')
ax.set_zlabel('Comprimento da Pétala (cm)')
ax.set_title('Centroides Resultantes')
plt.show()
