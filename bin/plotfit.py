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
            if ("S" in a) and (".fit" in a):
                a = open(a)
                for l in a:
                    for el in l.split():
                        if found:
                            averagen += 1
                            seeds.append(float(el))
                            found = False
                        if el == "bestgfit":
                            found = True
        named = os.getcwd()
        named = named.split("/")
        fname = named[len(named) - 1]
        data_dictionarie[fname] = seeds[:]
        seeds.clear()
        os.chdir(cpath) 

df = pd.DataFrame(data_dictionarie)

if averagen > 0:
    name = cpath
    name = name.split("/")
    dname = name[len(name) - 2]
    # sns.set_theme(style="ticks")
    ax = sns.boxplot(data = df, palette="vlag")
    ax.set(xlabel = "Algoritmo",ylabel= "Fitness generalizada", title = dname)
    plt.savefig("mygraph.png")
    plt.show() 
    

else:
    print("No data found")