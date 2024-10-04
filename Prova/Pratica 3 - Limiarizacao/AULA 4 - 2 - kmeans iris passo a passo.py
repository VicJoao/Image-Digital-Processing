import math
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def distancia_euclidiana(ponto1, ponto2):
    soma_quadrados = sum([(ponto1[i] - ponto2[i]) ** 2 for i in range(len(ponto1))])
    return math.sqrt(soma_quadrados)


def inicializar_centroides(dataset, k):
    centroides = random.sample(dataset, k)
    return centroides


def criar_clusters(dataset, centroides):
    clusters = [[] for _ in range(len(centroides))]
    for ponto in dataset:
        distancia_minima = float('inf')
        cluster_atual = None
        for i, centroide in enumerate(centroides):
            distancia = distancia_euclidiana(ponto, centroide)
            if distancia < distancia_minima:
                distancia_minima = distancia
                cluster_atual = i
        clusters[cluster_atual].append(ponto)
    return clusters


def calcular_media(cluster):
    num_atributos = len(cluster[0])
    media = [0] * num_atributos
    for ponto in cluster:
        for i in range(num_atributos):
            media[i] += ponto[i]
    media = [m / len(cluster) for m in media]
    return media


def atualizar_centroides(clusters):
    centroides = [calcular_media(cluster) for cluster in clusters]

    # Plota os dados
    plt.clf()
    cores = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(len(clusters)):
        for ponto in clusters[i]:
            plt.scatter(ponto[0], ponto[1], c=cores[i])
        if len(clusters[i]) > 0:
            plt.scatter(centroides[i][0], centroides[i][1], c=cores[-i - 1], marker='x')
    plt.pause(1)
    print("1")

    return centroides


def diferenca_centroides(centroides1, centroides2):
    if centroides1 is None or centroides2 is None:
        return float('inf')
    return sum([distancia_euclidiana(centroides1[i], centroides2[i]) for i in range(len(centroides1))])


def kmeans(dataset, k):
    centroides = inicializar_centroides(dataset, k)
    centroides_antigos = None
    while centroides_antigos is None or diferenca_centroides(centroides, centroides_antigos) > 0:
        clusters = criar_clusters(dataset, centroides)
        centroides_antigos = centroides
        centroides = atualizar_centroides(clusters)
        diferenca_centroides(centroides, centroides_antigos) if centroides_antigos is not None else float(
            'inf')
    return clusters


# Carrega o dataset Iris
X = load_iris()
iris = X.data.tolist()


# Executa o algoritmo K-Means no dataset Iris
k = 3
centroides = kmeans(iris, k)
