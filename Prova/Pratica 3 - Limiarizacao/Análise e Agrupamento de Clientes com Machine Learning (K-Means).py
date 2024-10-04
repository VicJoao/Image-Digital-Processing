#!/usr/bin/env python
# coding: utf-8

# ### Análise e Agrupamento de Clientes com Machine Learning (K-Means)
# https://minerandodados.com.br/analise-e-agrupamento-de-clientes-com-machine-learning-k-means/

# In[4]:


# Importando Bibliotecas:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import warnings
import os
warnings.filterwarnings("ignore")
py.offline.init_notebook_mode(connected = True)


#imports para svm

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score


# In[5]:


# Carregando a base de dados:
df = pd.read_csv('https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv')
# Verificando os dados:
df.columns=['id','genero','idade','Renda','Pontuação']
df.head()


# In[6]:


df.shape
(200, 5)
# Estatística Descritiva:
df.describe()


# In[7]:


# Tipos de Dados:
df.dtypes


# In[8]:


# Verificando registros nulos:
df.isnull().sum()


# In[9]:


# Definindo um estilo para os gráficos:
plt.style.use('fivethirtyeight')
# Verificando as distribuição dos dados:
plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in ['idade' , 'Renda' , 'Pontuação']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.distplot(df[x] , bins = 25)
    plt.title('{} '.format(x))
#plt.show()


# In[10]:


# Contagem de Amostras por Sexo:
plt.figure(1 , figsize = (15 , 5))
sns.countplot(y = 'genero' , data = df)
#plt.show()


# In[11]:


# Idade vs Renda Anual:
plt.figure(1 , figsize = (15 , 6))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'idade' , y = 'Renda' , data = df[df['genero'] == gender] ,
                s = 200 , alpha = 0.5 , label = gender)
plt.xlabel('idade'), plt.ylabel('Renda') 
plt.title('Idade vs Renda Anual')
plt.legend()
#plt.show()


# In[12]:


# Renda Anual vs Pontuação de Gastos:
plt.figure(1 , figsize = (15 , 6))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Renda',y = 'Pontuação' ,
                data = df[df['genero'] == gender] ,s = 200 , alpha = 0.5 , label = gender)
plt.xlabel('Renda'), plt.ylabel('Pontuação)') 
plt.title('Renda Anual vs Pontuação de Gastos')
plt.legend()
#plt.show()


# In[13]:


# Distribuição de Idade, Renda Anual e Pontuação de Gastos segmentado por Sexo:
plt.figure(1 , figsize = (15 , 7))
n = 0 
for cols in ['idade' , 'Renda' , 'Pontuação']:
    n += 1 
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.violinplot(x = cols , y = 'genero' , data = df , palette = 'vlag')
    sns.swarmplot(x = cols , y = 'genero' , data = df)
    plt.ylabel('Genero' if n == 1 else '')
    plt.title('Idade, Renda Anual e Pontuação de Gastos por Sexo' if n == 2 else '')
#plt.show()


# ### Agrupamento de dados utilizando o K-Means
# ## Algoritmo KMeans
# 
# n_clusters: número de clusters que queremos gerar com os nossos dados .
# init: se refere ao modo como o algoritmo será inicializado. k-means++: É o método padrão, e os centroides serão gerados utilizando um método inteligente que favorece a convergência. random: Se refere ao modo de inicialização de forma aleatória, ou seja, os centroides iniciais serão gerados de forma totalmente aleatória sem um critério para seleção. ndarray: array de valores indicando qual seriam os centroides que o algoritmo deveria utilizar para a inicialização .
# max_iter: Quantidade máxima de vezes que o algoritmo irá executar, por padrão o valor é 300 iterações.
# n_jobs: Quantos CPU´s iremos utilizar para executar o K-means.
# algorithm: Versão do algoritmo K-Means a ser utilizada. A versão clássica é executada através do valor full.
# Atributos Importantes
# 
# inertia: Soma das distâncias quadráticas intra cluster.
# 
# labels_: Rótulos dos Clusters atribuídos.
# 
# cluster_centers_: Valores dos Centroides.
# 
# ### Método Elbow
# O método Elbow é uma das formas usadas para descobrir a quantidade ideal de clusters no conjunto de dados.

# In[14]:


# Selecionando o número de clusters através do método Elbow (Soma das distâncias quadráticas intra clusters):
X2 = df[['Renda' , 'Pontuação']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n))
    algorithm.fit(X2)
    inertia.append(algorithm.inertia_)


# In[15]:


plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Número de Clusters') , plt.ylabel('Soma das Distâncias Q intra Clusters')
#plt.show()


# Conforme o número de clusters aumenta a soma das distâncias quadráticas intra clusters diminui, quando a diferença entre a distância é quase insignificante temos o valor ótimo de k , no nosso exemplo esse valor seria igual a 4.

# In[16]:


import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X2, method = 'ward'))
plt.title('Dendrogam', fontsize = 20)
plt.xlabel('Customers')
plt.ylabel('Ecuclidean Distance')
plt.rcParams['figure.figsize'] = (7,11)
#plt.show()


# In[17]:


# Inicializando e Computando o KMeans com o valor de 4 clusters:
algorithm = (KMeans(n_clusters = 4))
algorithm.fit(X2)

# Saída:

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,n_clusters=4, n_init=10, random_state=None, tol=0.0001, verbose=0)


# In[ ]:


# Visualizando os grupos criados e seus centroides:
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_

h = 0.02
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z2 = Z.reshape(xx.shape)
plt.imshow(Z2 , interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')
plt.scatter( x = 'Renda' ,y = 'Pontuação' , data = df , c = labels2 , s = 200 )
plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Pontuação de Gastos (1-100)') , plt.xlabel('Renda Anual (k$)')
plt.show()


# Solicitamos ao K-means que os dados fossem separados em 4 grupos e cada grupo têm o seu centroide correspondente (ponto em vermelho).
# 
# O centroide é o ponto de partida para cada cluster é a partir dele que todos os outros pontos fazem o cálculo de distância para definir a qual grupo o dado irá pertencer.
# 
# ### Analisando os dados agrupados

# In[ ]:


df["clusters"] = algorithm.labels_
df.head()



# In[ ]:


# Excluindo as colunas que não foram utilizadas:
df_group = df.drop(["id","idade"],axis=1).groupby("clusters")
# Estatística descritiva dos grupos:
df_group.describe()


# Conclusão:
# O K-means é uma técnica amplamente utilizada para fazer segmentação de clientes, essa estratégia pode ser usada em vários setores como campanhas de marketing, vendas, promoções etc, tudo vai depender do seu projeto.

# #Iniciando algoritmo SVM
# ##aproveitando os dados de Saída do Kmeans como dados 

# In[18]:


Y=labels2
Y


# In[19]:


#função para plotar os hiperplanos... recebe os valores de x e y e o classificador obtido
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')




# In[20]:


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
   
    if not title:
        if normalize:
            title = 'Matriz de Confusão normalizada'
        else:
            title = 'Matriz de Confusão NÃO normalizada'

    # calcula  a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusão normalizada")
    else:
        print('Matriz de Confusão NÃO normalizada')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ...criando os labels
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Entrada',
           xlabel='Saida')

    # rotacionando os labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Percorrendo sobre os dados e criando as anotações de texto
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



# In[21]:


#selecionando RENDA e PONTUAÇÃO apenas

X = df.iloc[:,[3,4]].values
plt.scatter(X[:,0], X[:,1], s = 30, c = 'blue', label = 'Clientes')

plt.xlabel('Renda')
plt.ylabel('PONTUAÇÃO')
plt.show()


# In[22]:


#criando o nome das 5 classes de saida
class_names = np.array(['Grupo1','Grupo2','Grupo3','Grupo4','Grupo5'])


# In[23]:


#dividindo o pacote de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.5, random_state=0)
#If train_size is also None, it will be set to 0.25. VER https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html


# In[24]:


#criando o modelo do classificador e obtendo as saidas para as entradas de teste
classificador = svm.SVC(kernel='linear', C=0.01)
y_pred = classificador.fit(X_train, y_train).predict(X_test)


# In[37]:


#convertendo para array, para aplicar na matriz de confusão
y_test=np.array(y_test)
y_pred=np.array(y_pred)


# In[38]:


np.set_printoptions(precision=2)
plot_decision_regions(X, Y, classifier=classificador)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# In[38]:





# In[41]:


# Plot normalized confusion matrix
#plot_confusion_matrix(y_test, y_pred, classes=class_names , normalize=True,
#                     title='Normalized confusion matrix')
plot_confusion_matrix(y_test, y_pred, classes=class_names , normalize=False,
                      title='Normalized confusion matrix')



# In[40]:


precisao=100*accuracy_score(y_test,y_pred)
print(f'Precisão do modelo: {precisao:.3f} %')


# # E agora, redes neurais MLP....
# # lembrando os dados
# X_train, X_test, y_train, y_test 

# In[42]:


import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import time
import timeit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# In[49]:


# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
classifier = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=5000,activation = 'relu',solver='adam',random_state=1)
classifier=classifier.fit(X_train, y_train)
#Using the trained network to predict

#Predicting y for X_val
y_pred = classifier.predict(X_test)
print("Precisão da rede MPL : ",accuracy_score(y_test, y_pred))
#print(cm)


# In[46]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_pred, y_test)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Valor true label')
plt.ylabel('saida rede (previsão)');


# In[ ]:




