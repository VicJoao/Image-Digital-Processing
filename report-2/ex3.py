# import numpy as np
# import cv2
#
# img = cv2.imread('amazonia1.jpg')
# Z = img.reshape((-1, 3))
# # Transformar a imagem em uma lista de tuplas (r, g, b)
# tuplas_unicas = set(map(tuple, Z))
# # Contar quantas tuplas únicas foram encontradas
# num_cor_pixel_utilizada = len(tuplas_unicas)
# print("Esta imagem utilizou %d combinações de cores RGB" % num_cor_pixel_utilizada)

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2

# Load image
image = cv2.imread('your_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape image data
reshaped_image = image.reshape(-1, 3)

# Define number of clusters
num_clusters = 5

# Create KMeans object
kmeans = KMeans(n_clusters=num_clusters)

# Fit KMeans to data
kmeans.fit(reshaped_image)

# Get cluster centers
cluster_centers = kmeans.cluster_centers_.astype(int)

# Plot clustered colors
plt.figure(figsize=(8, 6))
for color in cluster_centers:
    plt.plot([0, 1], [color, color], linewidth=10)
plt.axis('off')
plt.show()

# Assign labels to each pixel
labels = kmeans.labels_

# Reshape labels to original image shape
segmented_image = labels.reshape(image.shape[:2])

# Visualize segmented image
plt.imshow(segmented_image, cmap='viridis')
plt.axis('off')
plt.show()
