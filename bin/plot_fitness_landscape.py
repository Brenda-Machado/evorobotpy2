
"""
   This file belong to https://github.com/Brenda-Machado/evorobotpy
   and has been written by Brenda Silva Machado, brenda.silva.machado@grad.ufsc.br

   plot_fitness_landscape.py 

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('_mpl-gallery')

# Função principal para plotar os dados
def plot_data():
    if len(sys.argv) == 1:
        cpath = os.getcwd()
        files = os.listdir(cpath)
        print("Plotting data contained in:")
        
        all_stats = []
        count = 0
        
        for f in files:
            if f.endswith(".npy"):
                print(f)
                stat = np.load(f)
                size = np.shape(stat)
                newsize = (int(size[0] / 6), 6)
                stat = np.resize(stat, newsize)
                stat = np.transpose(stat)

                if count == 0:
                    shape = stat.shape
                    summed_stat = np.zeros(shape)
                    
                summed_stat += stat
                count += 1

        if count == 0:
            print("\033[1mERROR: No stat*.npy file found\033[0m")
            return
        
        # Calcula a média dos dados
        avg_stat = summed_stat / count
        print(avg_stat)
        
        # Prepare os dados para o gráfico 3D
        X = avg_stat[1]
        Y = avg_stat[1]
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.sin(R)

        # Criação do gráfico de superfície
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.Blues)
        
        # Adiciona título e rótulos aos eixos
        ax.set_title('Fitness Landscape')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show()

if __name__ == "__main__":
    plot_data()
