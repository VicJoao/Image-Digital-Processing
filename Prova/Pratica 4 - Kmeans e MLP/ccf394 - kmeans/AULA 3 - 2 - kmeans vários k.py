import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
Este código demonstra a aplicação do algoritmo de agrupamento K-means para quantização de cores em uma imagem. 
Primeiramente, a imagem é carregada em formato BGR. Em seguida, o algoritmo de K-means é aplicado com diferentes 
valores de k, representando o número de cores a serem utilizadas na quantização. Para cada valor de k, a imagem é 
recriada utilizando a paleta de cores resultante do algoritmo. As imagens resultantes são exibidas em uma única 
figura usando matplotlib, permitindo uma comparação visual dos efeitos da quantização de cores com diferentes números 
de clusters.
"""


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def color_quantization(image, k):
    """Performs color quantization using K-means clustering algorithm"""

    # Transform image into 'data':
    data = np.float32(image).reshape((-1, 3))
    # print(data.shape)

    # Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
    # In this case the maximum number of iterations is set to 20 and epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    # Apply K-means clustering algorithm:
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # At this point we can make the image with k colors
    # Convert center to uint8:
    center = np.uint8(center)
    # Replace pixel values with their center value:
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(16, 8))
plt.suptitle("Color quantization using K-means clustering algorithm", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load BGR image:
img = cv2.imread('landscape_1.jpg')

# Apply color quantization:
color_3 = color_quantization(img, 3)
color_5 = color_quantization(img, 5)
color_10 = color_quantization(img, 10)
color_20 = color_quantization(img, 20)
color_40 = color_quantization(img, 40)

# Plot the images:
show_img_with_matplotlib(img, "original image", 1)
show_img_with_matplotlib(color_3, "color quantization (k = 3)", 2)
show_img_with_matplotlib(color_5, "color quantization (k = 5)", 3)
show_img_with_matplotlib(color_10, "color quantization (k = 10)", 4)
show_img_with_matplotlib(color_20, "color quantization (k = 20)", 5)
show_img_with_matplotlib(color_40, "color quantization (k = 40)", 6)

# Show the Figure:
plt.show()
