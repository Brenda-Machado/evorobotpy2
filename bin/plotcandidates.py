#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file belongs to https://github.com/Brenda-Machado/evorobotpy2
   and has been written by Brenda Machado, brendamachado29016@gmail.com

   heatmap of the candidates
"""
print("")
print("plotcandidates.py")
print("heatmap of the candidates")
print("")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Ler o arquivo CSV contendo a lista de listas
nome_arquivo = "cma_es_candidates.csv"

# Definir o tamanho do lote
tamanho_lote = 100  # Você pode ajustar o tamanho do lote conforme necessário

# Inicialize uma matriz vazia para acumular os dados
matriz_acumulada = None

# Ler o arquivo CSV em lotes
data_chunks = pd.read_csv(nome_arquivo, chunksize=tamanho_lote)

# Processar cada lote e acumular os dados na matriz
for chunk in data_chunks:
    # Criar um DataFrame a partir do lote atual
    df = pd.DataFrame(chunk)

    # Converter o DataFrame em uma matriz NumPy
    matriz_lote = df.to_numpy()

    # Acumular os dados na matriz acumulada
    if matriz_acumulada is None:
        matriz_acumulada = matriz_lote
    else:
        matriz_acumulada = np.vstack((matriz_acumulada, matriz_lote))

# Gere o heatmap usando a matriz acumulada
df = pd.DataFrame(matriz_acumulada)
print(df)
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=False, cmap='YlGnBu', cbar=True)

# Configurações opcionais de título e rótulos dos eixos
plt.title('Heatmap das soluções candidatas do CMA-ES')
plt.ylabel('Gerações')
plt.xlabel('Parâmetros')

# Exibir o heatmap
plt.savefig("heatmap.png")
plt.show()