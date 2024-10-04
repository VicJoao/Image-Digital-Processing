import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def get_elbow(src, k_range):
    img = cv2.imread(src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pixels = img.reshape(-1, 3)
    
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(pixels)
        sse.append(kmeans.inertia_)
    
    plt.plot(k_range, sse, marker='o')
    plt.xlabel('Numero de clusters (k)')
    plt.ylabel('Soma dos erros ao quadrado (SSE)')
    plt.title('Método Elbow')
    plt.show()

    return sse

def get_colors(src, k):
    img = cv2.imread(src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pixels = img.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_
    
    return colors.astype(int), pixels

def plot_colors(colors):
    plt.figure(figsize=(8, 6))
    plt.pie(np.ones(len(colors)), colors=colors / 255, startangle=90)
    plt.axis('equal')
    plt.show()

def plot_pixels(pixels):
    fig = plt.figure(figsize=(8, 6))
    axis = fig.add_subplot(111, projection='3d')

    axis.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], c='blue', marker='.', s=100)

    axis.set_xlabel('Red')
    axis.set_ylabel('Green')
    axis.set_zlabel('Blue')

    plt.show()

img = "halteres.jpg"

colors, pixels = get_colors(img, 7)
num_colors = len(np.unique(pixels, axis=0))

print("\nCentroides (RGB):\n", colors)
plot_colors(colors)

k_range = range(3, 11)
sse = get_elbow(img, k_range)

elbow_index = np.argmin(np.diff(sse, 2)) + 1
elbow = k_range[elbow_index]
print("\n\nPonto de inflexão (k):", elbow)

colors_k_minus_1 = get_colors(img, elbow - 1)[0]
colors_k = get_colors(img, elbow)[0]
colors_k_plus_1 = get_colors(img, elbow + 1)[0]
colors_k_plus_2 = get_colors(img, elbow + 2)[0]

print("\nCentroides k-1:\n", colors_k_minus_1)
print("\nCentroides k:\n", colors_k)
print("\nCentroides k+1:\n", colors_k_plus_1)
print("\nCentroides k+2:\n", colors_k_plus_2, "\n\n")
plot_colors(colors_k_minus_1)
plot_colors(colors_k)
plot_colors(colors_k_plus_1)
plot_colors(colors_k_plus_2)

print("Número de cores (RGB):", num_colors, "\n\n")
plot_pixels(pixels)
