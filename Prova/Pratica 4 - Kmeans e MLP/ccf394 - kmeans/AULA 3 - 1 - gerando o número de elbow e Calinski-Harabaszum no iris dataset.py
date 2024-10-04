import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import calinski_harabasz_score

"""
O código realiza uma análise de agrupamento (clustering) utilizando o conjunto de dados Iris. Primeiramente, 
é calculada a inércia para diferentes números de clusters, e o método Elbow é aplicado para identificar o número 
ótimo de clusters com base no ponto de inflexão da curva da inércia. Em seguida, é utilizado o Método da 
Calinski-Harabasz para calcular o índice de variação entre clusters para diferentes números de clusters. O número 
ótimo de clusters é determinado pelo valor que maximiza esse índice. Ambos os métodos fornecem uma abordagem para 
determinar o número ideal de clusters em uma análise de agrupamento, com o objetivo de encontrar uma estrutura de 
cluster significativa nos dados.
"""


# Passo 1: Carregar o conjunto de dados Iris
iris_data = load_iris()
X = iris_data.data  # Recursos

# Calcular a inércia para diferentes números de clusters
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
# Encontrar o número de clusters usando o KneeLocator
kneedle = KneeLocator(range(1, 11), inertias, curve='convex', direction='decreasing')
elbow_number = kneedle.elbow

print("Número de clusters no elbow:", elbow_number)

# Plotar o gráfico da inércia em relação ao número de clusters
plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('Inércia')
plt.title('Método Elbow para o Dataset Iris')
plt.xticks(range(1, 11))
plt.show()

# agora, utilizando Método da Calinski-Harabasz (índice de variação entre clusters)
# Este método calcula a razão entre a dispersão dentro dos clusters e a dispersão entre os clusters.
# Um valor mais alto indica clusters mais densos e bem separados.
# O número ideal de clusters é aquele que maximiza este índice.


# Calcular o índice de variação entre clusters para diferentes números de clusters
scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    score = calinski_harabasz_score(X, kmeans.labels_)
    scores.append(score)

# Encontrar o número de clusters com o maior índice de variação entre clusters
optimal_num_clusters = np.argmax(scores) + 2  # +2 porque começamos a partir de k=2

print("Número ótimo de clusters pela Calinski-Harabasz:", optimal_num_clusters)
