import numpy as np
import pandas as pd
import plotly.graph_objs as go
from PIL import Image
from sklearn.cluster import KMeans

"""
O código fornecido realiza a segmentação de uma imagem RGB usando o algoritmo K-Means e plotando a distribuição 
dos pixels no espaço tridimensional RGB antes e depois do agrupamento. Primeiro, a imagem é carregada e convertida em 
um array numpy, onde os valores de R, G e B de cada pixel são obtidos. Em seguida, o algoritmo K-Means é aplicado aos 
valores RGB dos pixels, agrupando-os em um número especificado de clusters. Os centróides dos clusters são calculados 
e plotados como esferas pretas nos gráficos 3D interativos, representando o espaço de cores antes e depois do 
agrupamento. Os pontos de dados (pixels) são representados como marcadores coloridos no espaço RGB. Além disso, 
os valores dos centróides são exibidos na saída. O usuário pode inserir o número desejado de clusters quando 
solicitado.
"""


def plot_rgb_distribution(image_path, num_clusters):
    # Carrega a imagem RGB
    img = Image.open(image_path)

    # Converte a imagem para um array numpy
    img_array = np.array(img)

    # Obtém os valores R, G e B de cada pixel
    r_values = img_array[:, :, 0].flatten()
    g_values = img_array[:, :, 1].flatten()
    b_values = img_array[:, :, 2].flatten()

    # Cria o dataframe com os valores RGB
    data = {'Red': r_values, 'Green': g_values, 'Blue': b_values}
    df = pd.DataFrame(data)

    # Aplica o algoritmo K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    df['Cluster'] = kmeans.fit_predict(df)

    # Obtém os centróides dos clusters
    centroids = kmeans.cluster_centers_

    # Cria o gráfico 3D interativo antes do agrupamento
    fig_before = go.Figure(data=[go.Scatter3d(
        x=df['Red'],
        y=df['Green'],
        z=df['Blue'],
        mode='markers',
        marker=dict(
            size=3,
            color='rgb(255, 0, 0)',  # define a cor dos marcadores
            opacity=0.8
        ),
        name='Antes do agrupamento'
    )])

    # Adiciona as esferas dos centróides ao gráfico antes do agrupamento
    for centroid in centroids:
        fig_before.add_trace(go.Scatter3d(
            x=[centroid[0]],
            y=[centroid[1]],
            z=[centroid[2]],
            mode='markers',
            marker=dict(
                size=5,
                color='black',  # define a cor das esferas dos centróides
                opacity=0.8,
                symbol='circle'
            ),
            showlegend=False
        ))

    # Define o layout do gráfico antes do agrupamento
    fig_before.update_layout(scene=dict(
        xaxis_title='Red',
        yaxis_title='Green',
        zaxis_title='Blue',
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    ))

    # Plota o gráfico interativo antes do agrupamento
    fig_before.show()

    # Cria o gráfico 3D interativo depois do agrupamento
    fig_after = go.Figure()
    for i in range(num_clusters):
        cluster_points = df[df['Cluster'] == i]
        fig_after.add_trace(go.Scatter3d(
            x=cluster_points['Red'],
            y=cluster_points['Green'],
            z=cluster_points['Blue'],
            mode='markers',
            marker=dict(
                size=3,
                opacity=0.8
            ),
            name=f'Cluster {i}'
        ))

    # Adiciona as esferas dos centróides ao gráfico depois do agrupamento
    for centroid in centroids:
        fig_after.add_trace(go.Scatter3d(
            x=[centroid[0]],
            y=[centroid[1]],
            z=[centroid[2]],
            mode='markers',
            marker=dict(
                size=5,
                color='black',  # define a cor das esferas dos centróides
                opacity=0.8,
                symbol='circle'
            ),
            showlegend=False
        ))

    # Define o layout do gráfico depois do agrupamento
    fig_after.update_layout(scene=dict(
        xaxis_title='Red',
        yaxis_title='Green',
        zaxis_title='Blue',
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    ))

    # Plota o gráfico interativo depois do agrupamento
    fig_after.show()

    # Mostra os valores dos centróides
    print("Valores dos centróides:")
    for i, (centroid) in enumerate(centroids):
        centroid = np.uint8(centroid)
        print(f"Cluster {i}: R={centroid[0]}, G={centroid[1]}, B={centroid[2]}")


# Caminho para a imagem RGB
image_path = 'capacete.jpg'

# Número de clusters desejados
num_clusters = int(input("Digite o número de clusters desejados: "))

# Chama a função para plotar a distribuição RGB com K-Means
plot_rgb_distribution(image_path, num_clusters)
