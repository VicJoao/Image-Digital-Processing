from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
"""
## Glass Dataset

A análise forense da composição do vidro pode revelar a proveniência de um pedaço de vidro,
no entanto, diferentes marcas de vidro são ligeiramente diferentes umas das outras.
Construiremos um modelo que classificará o vidro com base em sua composição,
e usaremos o aprendizado on-line para que estejamos prontos para aprender com novos dados a qualquer momento.
Se alguém nos pedir para ser detetives da polícia algum dia, agora temos uma vantagem.

Este é um conjunto de dados no repositório de aprendizado de máquina UCI, precisamos construir nossa função `load_glass`.
As classes são numeradas a partir de $1$ no conjunto de dados


Sem saber muito sobre esse modelo,
vamos tentar usar uma rede neural para classificar esse conjunto de dados.
Sabemos que a rede neural é uma boa quantidade
de perceptrons interconectados e que alteramos os pesos do perceptron
com base em erros de classificação para alcançar a convergência.

Este é um conjunto de dados real,
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import os
import sys
import requests
from sklearn import datasets
from sklearn.utils import Bunch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_glass():
    #inicialmente vai criar uma pasta scikit_learn_data dentro do C:\\users\\usiario
    glass_dir = 'uci_glass'
    data_dir = datasets.get_data_home()
    data_path = os.path.join(data_dir, glass_dir, 'glass.data')
    descr_path = os.path.join(data_dir, glass_dir, 'glass.names')
    glass_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
    glass_descr = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.names'
    os.makedirs(os.path.join(data_dir, glass_dir), exist_ok=True)
    print(os.path.join(data_dir, glass_dir))
    try:
        with open(descr_path, 'r') as f:
            descr = f.read()
    except IOError:
        print('Baixando dados de ....', glass_descr, file=sys.stderr)
        r = requests.get(glass_descr)
        with open(descr_path, 'w') as f:
            f.write(r.text)
        descr = r.text
        r.close()
    try:
        data = pd.read_csv(data_path, delimiter=',', header=None).values
        # vamos gravar o array em disco apenas para verificar
        np.savetxt("vidros_dataset.csv", data, delimiter=";")
        
    except IOError:
        print('Baixando dados de ....', glass_data, file=sys.stderr)
        r = requests.get(glass_data)
        with open(data_path, 'w') as f:
            f.write(r.text)
        r.close()
        data = pd.read_csv(data_path, delimiter=',', header=None).values
    target = data[:, 10].astype(np.uint8).copy()
    target[target > 3] -= 1  # fix non-existent classes
    target -= 1              # fix class numbering
    return Bunch(DESCR=descr,
                 data=data[:, :10].copy(),
                 feature_names=['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'],
                 target=target,
                 target_names=['windows_float_processed',
                               'windows_non_float_processed',
                               'vehicle_windows',
                               'containers',
                               'tableware',
                               'headlamps'])


glass = load_glass()
print(glass.DESCR)
print(pd.Series(glass.target).value_counts())

      

"""
Sem saber muito sobre esse modelo,
vamos tentar usar uma rede neural para classificar esse conjunto de dados.
Sabemos que a rede neural é uma boa quantidade
de perceptrons interconectados e que alteramos os pesos do perceptron
com base em erros de classificação para alcançar a convergência.

Este é um conjunto de dados real,

Uma coisa que podemos fazer é selecionar um bom subconjunto de classes e dividir os
conjuntos de treinamento e teste a partir deles.
Mas tentaremos usar o conjunto de dados completo,
passamos `stratify=` para `train_test_split` que desta forma tentará
para manter o mesmo suporte de classe nos conjuntos de treinamento e teste.


"""

X_treino, X_teste, y_treino, y_teste  = train_test_split(glass.data, glass.target, test_size=0.2, stratify=glass.target)


# Dividindo em conjuntos de treinamento e teste
#X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# Definindo os parâmetros para GridSearchCV
# vai executar todas as combinações possiveis entre os parametros abaixo e achar a melhor solução
parametros = {'hidden_layer_sizes': [(50,), (100,), (50,50), (100,100)],
              'alpha': [0.01, 0.001],
              'learning_rate_init': [0.01, 0.001],
              'max_iter':[100, 1000, 5000]}


# Criando o GridSearchCV
mlp = MLPClassifier( solver='adam', random_state=42)
"""
O GridSearchCV é uma ferramenta usada para automatizar o processo de ajuste dos parâmetros de um algoritmo, 
pois ele fará de maneira sistemática diversas combinações dos parâmetros
 e depois de avaliá-los os armazenará num único objeto.

O GridSearchCV é um módulo do Scikit Learn e é amplamente usado para automatizar grande parte do processo de tuning. 
O objetivo primário do GridSearchCV é a criação de combinações de parâmetros para posteriormente avaliá-las.
"""

grid = GridSearchCV(mlp, parametros, cv=5, verbose=2, n_jobs=-1)
grid.fit(X_treino, y_treino)

# Imprimindo os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros: ", grid.best_params_)
"""
A validação cruzada é uma técnica utilizada para avaliar o desempenho de um modelo de aprendizado de máquina. Em vez de simplesmente dividir o conjunto de dados em conjuntos de treinamento e teste, a validação cruzada divide o conjunto de dados em k folds (partes) diferentes, onde k é um número inteiro pré-definido.

Em seguida, o modelo é treinado k vezes, em que cada uma das k partes é usada como conjunto de teste uma vez, enquanto as outras k-1 partes são usadas como conjunto de treinamento. Ou seja, em cada uma das k vezes, o modelo é treinado em uma combinação diferente de conjuntos de treinamento e teste.

Ao final do processo, o desempenho do modelo é avaliado pela média dos resultados obtidos nas k execuções do modelo. Isso permite obter uma avaliação mais confiável do desempenho do modelo, pois ele é avaliado em k combinações diferentes de conjuntos de treinamento e teste.

A validação cruzada é particularmente útil quando o conjunto de dados é pequeno,
pois permite avaliar o desempenho do modelo usando todos os dados disponíveis
para treinamento e teste. Além disso, a validação cruzada pode ajudar a
evitar o overfitting (sobreajuste) do modelo, pois o modelo é avaliado em
várias combinações diferentes de treinamento e teste, e não apenas em uma
única divisão do conjunto de dados.



"""
# Avaliando o modelo usando validação cruzada

"""
Alguns modelos podem não convergir, mas a validação cruzada deve eliminá-los.
E o melhor modelo deve ser uma rede convergente.


"""
cv_score = cross_val_score(grid.best_estimator_, X_treino, y_treino, cv=5)
print("Precisão  de validação cruzada: ", np.mean(cv_score)*100)

# Avaliando o modelo no conjunto de teste
y_pred = grid.predict(X_teste)

# Plotando a matriz de confusão de forma gráfica
cm = confusion_matrix(y_teste, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,  display_labels=glass.target_names)
fig, ax = plt.subplots(figsize=(20,20))
disp.plot(ax=ax)
plt.title("Matriz de Confusão")
plt.xlabel("Predições")
plt.ylabel("Valores Verdadeiros")
plt.show()
