import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função para carregar todos os arquivos npy de uma pasta
def carregar_arquivos_stat(pasta):
    arquivos_stat = [os.path.join(pasta, f) for f in os.listdir(pasta) if f.endswith('.npy')]
    dados = [np.load(arq).reshape(-1, 6) for arq in arquivos_stat]
    return dados

# Função para truncar ou preencher os dados para que todos tenham o mesmo tamanho
def ajustar_dados(dados):
    # Encontrar o menor número de linhas entre os arquivos
    tamanho_minimo = min([d.shape[0] for d in dados])
    
    # Truncar ou ajustar todos os arquivos para o tamanho mínimo
    dados_ajustados = [d[:tamanho_minimo, :] for d in dados]
    return dados_ajustados

# Função para calcular a média dos dados de múltiplos arquivos
def calcular_media_dados(dados):
    dados_empilhados = np.stack(dados, axis=0)
    media_dados = np.mean(dados_empilhados, axis=0)
    return media_dados

# Função para gerar a paisagem de fitness
def gerar_paisagem_fitness(media_dados):
    # Separando os dados correspondentes
    avgfit = media_dados[:, 4]  # Eixo Z
    center = media_dados[:, 5]  # Eixo X
    print(center)

    gerar_histograma_2d(center, avgfit)
    
    n_points = int(np.sqrt(len(center)))  # Definir o número de pontos para a malha

    # # Redimensionar steps e bfit para formar uma malha compatível
    center_reshaped = center[:n_points**2].reshape(n_points, n_points)
    print(center_reshaped)
    avgfit = avgfit[:n_points**2].reshape(n_points, n_points)

    X, Y = np.meshgrid(center_reshaped[0], center_reshaped[0])

    # Plotando o gráfico
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Superfície 3D com estilo e limites ajustados
    ax.plot_surface(X, Y, avgfit, cmap='Blues', edgecolor='none')

    # Rótulos dos eixos
    ax.set_title('Fitness Landscape')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # ax.set_xlim(center.min(), center.max())
    # ax.set_ylim(center.min(), center.max())
    # ax.set_zlim(avgfit.min(), avgfit.max())

    # Mostrando o gráfico
    plt.show()

# Função para gerar histogramas de center e avgfit
def gerar_histograma_2d(center, avgfit):
    plt.figure(figsize=(10, 6))
    plt.hist2d(center, avgfit, bins=30, cmap='Blues', cmin=1)
    plt.colorbar(label='Frequência')
    
    # Configurações do gráfico
    plt.title('Histograma 2D de Center vs. Average Fitness')
    plt.xlabel('Center')
    plt.ylabel('Average Fitness')
    plt.tight_layout()
    plt.show()

# Função principal para o pipeline completo
def main(pasta_dados):
    # Carregar dados de todos os arquivos na pasta
    dados = carregar_arquivos_stat(pasta_dados)
    
    # Ajustar os dados para ter o mesmo tamanho
    dados_ajustados = ajustar_dados(dados)
    
    # Calcular a média dos dados
    media_dados = calcular_media_dados(dados_ajustados)
    
    # Gerar a paisagem de fitness
    gerar_paisagem_fitness(media_dados)

# Exemplo de uso
# pasta_dados = 'episodes-5'
pasta_dados = 'maxsteps-100'
main(pasta_dados)
