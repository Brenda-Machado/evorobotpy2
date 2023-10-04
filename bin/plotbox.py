#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file belongs to https://github.com/Brenda-Machado/evorobotpy2
   and has been written by Brenda Machado, brendamachado29016@gmail.com

   boxplot of the final results of each condition
"""
print("")
print("plotfit.py")
print("boxplot of the final results of each condition")
print("")

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
averagen = 0
found = False
seeds = []
data_dictionarie = {}

if len(sys.argv) == 1:
    # sns.set_theme()
    cpath = os.getcwd()
    folders = os.listdir(cpath)
    for f in folders:
        os.chdir(f)
        tpath = os.getcwd() 
        files = os.listdir(tpath)
        for a in files:
            if (".csv" in a):
                a = pd.read_csv(a)
                a = a.to_numpy()
                temp = []
                for l in a:
                    temp.append(float(l))
                seeds.append(np.mean(temp))
            averagen += 1
        named = os.getcwd()
        named = named.split("/")
        fname = named[len(named) - 1]  
        data_dictionarie[fname] = seeds[:]
        seeds = []
        averagen = 0 
        os.chdir(cpath) 

df = pd.DataFrame(data_dictionarie)
print(df)

averagen += 1

if averagen > 0:
    name = cpath
    name = name.split("/")
    dname = name[len(name) - 2]
    ax = sns.boxplot(data = df, palette="vlag", hue_order = ["OpenAI-ES", "OpenAI-ES-NE", "SSS", "SSS-NE"], order = ["OpenAI-ES", "OpenAI-ES-NE", "SSS", "SSS-NE"])
    ax.set(xlabel = "Algoritmo",ylabel= "Fitness generalizada", title = dname)
    plt.savefig("mygraph.png")
    plt.show() 
    
else:
    print("No data found")