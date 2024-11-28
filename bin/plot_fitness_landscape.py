import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Carregar o dicionário do arquivo .npy
dados = np.load('state_reward_exp_1.npy', allow_pickle=True).item()

# Extrair as coordenadas (x, y) e os valores de fitness
coordenadas = list(dados.keys())
fitness = list(dados.values())

# Converter as coordenadas (x, y) para arrays separados
x_vals = np.array([coord[0] for coord in coordenadas])
y_vals = np.array([coord[1] for coord in coordenadas])

# Verificar se há valores NaN ou infinitos nos dados de fitness
fitness = np.array(fitness)

# Definir uma grade regular para interpolação
x_grid = np.linspace(np.min(x_vals), np.max(x_vals), len(x_vals))
y_grid = np.linspace(np.min(y_vals), np.max(y_vals), len(y_vals))
x_grid, y_grid = np.meshgrid(x_grid, y_grid)

# Interpolação dos valores de fitness para preencher a grade usando o método 'linear'
fitness_grid = griddata((x_vals, y_vals), fitness, (x_grid, y_grid), method='linear')

# Criar o gráfico 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotar a superfície 3D com a grade interpolada
surf = ax.plot_surface(x_grid, y_grid, fitness_grid, cmap='Blues', edgecolor='none')

# Adicionar título e rótulos aos eixos
ax.set_title("Fitness Landscape")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Fitness")

# Limite máximo igual a 0 para o eixo Z
# ax.set_zlim(np.min(fitness), 0)

# Mostrar o gráfico
plt.show()
