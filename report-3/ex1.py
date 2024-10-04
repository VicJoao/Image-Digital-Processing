import cv2
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans

def read_image(filepath):
    img = cv2.imread(filepath)
    if img is None:
        print("Erro: Não foi possível carregar a imagem.")
        return None
    return img

def image_to_dataset(img):
    return img.reshape(-1, 3)

def plot_pixel_distribution_3d(dataset):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=dataset / 255, s=1)
    ax.set_xlabel('Vermelho')
    ax.set_ylabel('Verde')
    ax.set_zlabel('Azul')
    ax.set_title('Distribuição de Pixels')
    plt.show()

def find_optimal_k(dataset):
    distortions = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(dataset)
        distortions.append(kmeans.inertia_)

    knee = KneeLocator(list(K_range), distortions, curve='convex', direction='decreasing')
    optimal_k = knee.elbow

    return optimal_k

def kmeans_clustering(dataset, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(dataset)
    return labels

def print_cluster_labels(labels):
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    print("Número total de clusters:", len(unique_labels))
    print("Contagem de pontos em cada cluster:")
    for label, count in zip(unique_labels, label_counts):
        print(f"Cluster {label}: {count} pontos")

    plt.figure()
    plt.bar(unique_labels, label_counts)
    plt.xlabel('Cluster')
    plt.ylabel('Número de pontos')
    plt.title('Contagem de pontos em cada cluster')
    plt.show()



def main():
    img_path = "halteres.jpg"
    img = read_image(img_path)
    if img is not None:
        dataset = image_to_dataset(img)
        plot_pixel_distribution_3d(dataset)
        optimal_k = find_optimal_k(dataset)
        print("Número ótimo de clusters:", optimal_k)
        labels = kmeans_clustering(dataset, optimal_k)
        print_cluster_labels(labels)

if __name__ == "__main__":
    main()
