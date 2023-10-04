#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file belongs to https://github.com/arthurholtrup/evorobotpy2
   and has been written by Arthur H. Bianchini, arthur.h.bianchini@grad.ufsc.br

   plotcompare.py the fitness across generation contatined in stat*.npy files

"""


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

print("plotcompare.py")
print("plot the fitness across generation contatined in stat*.npy files")
print("if called with a filename, plot the data of that file")
print("if called without parameters, plot the data of all available stat*.npy files")
print("")

# plot the data contained in the parameter file
# if called without parameters plot all available statS?.npy files

statsumn = 0
statavesum = 0
np.random.seed(1)
ave_sss = []
ave_sss_ne = []

if len(sys.argv) == 1:
    cpath = os.getcwd()
    files = os.listdir(cpath)
    print("Plotting data contained in:")
    stats = []
    for f in files:
        if "statS" in f:
            stats.append(f)
    # minmax = [[None, None] for _ in range((len(stats)))]
    # print(minmax)
    # print(minmax)
    count = 0
    for f in files:
        if "statS" in f:
            print(f)
            fint = f.replace("statS", '')
            fint = fint.replace(".npy", '')
            stat = np.load(f, allow_pickle=True)
            size = np.shape(stat)
            newsize = (int(size[0]), 6)
            if 10 < int(fint) <= 20:
                newsize = (int(size[0]), 4)
            stat = np.resize(stat, newsize)
            stat = pd.DataFrame(stat)
            if int(fint) < 11:
                for i in range(len(stat[0])):
                    ave_sss.append((stat[0][i], stat[2][i]))
            else:
                for i in range(len(stat[0])):
                    ave_sss_ne.append((stat[0][i], stat[2][i]))
            statavesum += 1
            statsumn = statsumn + 1
            count += 1
    
    # ave_es = []
    # ave_es_ne = []

    # for f in stats:
    #     fint = f.replace("statS", '')
    #     fint = fint.replace(".npy", '')
    #     fint = int(fint)
    #     stat = np.load(f, allow_pickle=True)
    #     size = np.shape(stat)
    #     newsize = (int(size[0]), 6)
    #     if 10 < fint <= 20:
    #        newsize = (int(size[0]), 4)
    #     stat = np.resize(stat, newsize)
    #     stat = pd.DataFrame(stat)
    #     print(stat)
    #     new_x = np.linspace(minmax[fint-1][0], minmax[fint-1][1], 100)
    #     new_y = np.interp(new_x, stat[0].astype(float), stat[2].astype(float))
    #     print(new_y)
    #     # new_y = np.interp(new_x, stat[:, 0].astype(float), stat[:, 2].astype(float))

    #     # if fint < 11:
    #     #     ave_es.append([new_x, new_y])
    #     # elif fint < 21:
    #     #     ave_es_ne.append([new_x, new_y])
    #     # elif fint < 31:
    #     #     ave_sss.append([new_x, new_y])
    #     # elif fint < 41:
    #     #     ave_sss_ne.append([new_x, new_y])
    #     if fint < 11:
    #         ave_sss.append([new_x, new_y])
    #     elif fint < 21:
    #         ave_sss_ne.append([new_x, new_y])

    # # aves = [ave_es, ave_es_ne, ave_sss, ave_sss_ne]
    # # aves = [ave_es, ave_es_ne]
    aves = [ave_sss, ave_sss_ne]
    # # label = ["OpenAI-ES","OpenAI-ES-NE", "SSS", "SSS-NE"]
    # # label = ["OpenAI-ES","OpenAI-ES-NE"]
    # label = ["SSS", "SSS-NE"]
    # count = 0
    # for ave in aves:
    #     midx = [np.mean([ave[file][0][j] for file in range(len(ave[:]))]) for j in range(100)]
    #     midy = [np.mean([ave[file][1][j] for file in range(len(ave[:]))]) for j in range(100)]
    #     plt.plot(midx, midy, label = label[count])
    #     count += 1
    #     ci = 1.96 * np.std(midy)/np.sqrt(len(midx))
    #     plt.fill_between(midx, (midy-ci), (midy+ci), alpha=0.1)

    for ave in aves:
        plt.plot(ave)
    
    
    plt.xlabel("Steps")
    plt.ylabel("Generalized fitness")
    plt.title(input("Insert plot title: "))
    plt.legend()
    plt.savefig("mygraph.png")
    plt.show()

    if statsumn == 0:
        print("\033[1mERROR: No stat*.npy file found\033[0m")
