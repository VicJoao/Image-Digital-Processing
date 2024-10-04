import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class_colors = {
    0: 'blue',
    1: 'black',
    2: 'yellow',
    3: 'red',
    4: 'grey',
    5: 'green',
    6: 'purple'
}

df = pd.read_csv('dados_ex4.csv', header=None, names=['Blue', 'Green', 'Red', 'Class'])

# Separate data by class
classes = df['Class'].unique()

# Plot data for each class
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for c in classes:
    class_data = df[df['Class'] == c][['Blue', 'Green', 'Red']]
    color = class_colors[c]
    ax.scatter(class_data['Blue'], class_data['Green'], class_data['Red'], c=color)

ax.set_xlabel('Blue')
ax.set_ylabel('Green')
ax.set_zlabel('Red')

plt.show()
