import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
Este código realiza a correspondência de histograma entre duas imagens em escala de cinza. Ele ajusta os valores 
de pixel de uma imagem de origem para que seu histograma corresponda ao histograma de uma imagem de modelo (template).

Primeiro, a função hist_match é definida para realizar a correspondência de histograma. Em seguida, a função ecdf é 
definida para calcular a função de distribuição cumulativa empírica (ECDF) de uma imagem.

Depois, a imagem de origem e a imagem de modelo são lidas usando OpenCV. A função hist_match é aplicada para 
corresponder o histograma da imagem de origem ao histograma da imagem de modelo.

Finalmente, os resultados são visualizados usando matplotlib. As três imagens - a imagem de origem, a imagem de 
modelo e a imagem correspondida - são exibidas em três subplots separados. Além disso, os gráficos da ECDF são 
plotados para cada imagem para mostrar a distribuição cumulativa dos valores de pixel.
"""


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Args:
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened array
        template: np.ndarray
            Template image; can have different dimensions to source

    Returns:
        matched: np.ndarray
            The transformed output image
    """
    # Flatten the images
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # Compute the histograms and cumulative distribution functions
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64) / s_counts.sum()
    t_quantiles = np.cumsum(t_counts).astype(np.float64) / t_counts.sum()

    # Interpolate to find the pixel values in the template image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def ecdf(x):
    """Compute the empirical cumulative distribution function"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64) / counts.sum()
    return vals, ecdf


# Read the source and template images
source = cv2.imread("lena.png", 0)
template = cv2.imread("ascent.png", 0)

# Perform histogram matching
matched = hist_match(source, template)

# Compute ECDF for each image
x1, y1 = ecdf(source.ravel())
x2, y2 = ecdf(template.ravel())
x3, y3 = ecdf(matched.ravel())

# Plot the images and ECDFs
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax in axes.flatten():
    ax.axis("off")

axes[0, 0].imshow(source, cmap=plt.cm.gray)
axes[0, 0].set_title("Source")
axes[0, 1].imshow(template, cmap=plt.cm.gray)
axes[0, 1].set_title("Template")
axes[0, 2].imshow(matched, cmap=plt.cm.gray)
axes[0, 2].set_title("Matched")

axes[1, 0].plot(x1, y1 * 100, "-r", lw=3, label="Source")
axes[1, 0].plot(x2, y2 * 100, "-k", lw=3, label="Template")
axes[1, 0].plot(x3, y3 * 100, "--r", lw=3, label="Matched")
axes[1, 0].set_xlim(x1[0], x1[-1])
axes[1, 0].set_xlabel("Pixel value")
axes[1, 0].set_ylabel("Cumulative %")
axes[1, 0].legend(loc="best")

plt.tight_layout()
plt.show()
