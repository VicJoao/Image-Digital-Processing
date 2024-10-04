from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

"""
Este código realiza a análise de agrupamento K-Means em um conjunto de dados Iris para determinar o número ideal 
de clusters. Primeiro, ele carrega o conjunto de dados Iris e extrai as características das flores. Em seguida, 
ele itera sobre um intervalo de possíveis números de clusters (de 1 a 10) e calcula a distorção para cada 
configuração de cluster. A distorção é uma medida da soma dos quadrados das distâncias entre cada ponto de dado e o 
centróide do cluster mais próximo. Essa métrica é representada graficamente em um gráfico, onde o eixo x representa o 
número de clusters e o eixo y representa a distorção. O objetivo é identificar o ponto onde a distorção começa a 
diminuir menos rapidamente, indicando o número ideal de clusters. Neste caso, é onde a curva se estabiliza ou forma 
um "cotovelo". Finalmente, o modelo K-Means é treinado com o número ideal de clusters e os centróides dos clusters 
são impressos. Isso fornece informações sobre os centros de cada grupo identificado.
"""

# Carrega o conjunto de dados iris
iris = load_iris()
X = iris.data

# Cria um objeto k-means com inicialização k-means++
distortions = []
# Calcula a distorção para um intervalo de número de clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

# Plotagem da curva de distorção
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('Distorção')
plt.title('Método Elbow para Determinar o Número Ideal de Clusters')
plt.show()

# Treina o modelo com o número ideal de clusters
kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(X)

# Imprime os centros dos clusters encontrados
print("Centros dos Clusters:")
print(kmeans.cluster_centers_)
