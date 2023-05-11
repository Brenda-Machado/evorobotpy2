#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/Brenda-Machado/evorobotpy2
   and has been written by Brenda S. Machado, brenda.silva.machado@grad.ufsc.br
"""

""" TO-DO
[] Verificar se .csv vai ser ok de analisar pelo pandas
"""

import numpy as np
import pandas as pd
from numpy import zeros, ones, dot, sqrt

class Phylogenetic:
    def __init__(self, number_niches, max_gens, seed):
        # matrix com nan e linhas = max_gens e colunas = number_niches
        self.matrix_colonization = np.empty((number_niches, int(max_gens)))
        self.matrix_colonization[:] = np.nan
        self.seed = seed
        self.cgen = 0
        self.initial_state()

    def initial_state(self):
        for i in range((0,len(self.matrix_colonization))):
            # primeira coluna da matriz deve ser preenchida com o número do próprio nicho
            self.matrix_colonization[i][0] = i

    def colonize(self, colonizer, colonized):
        self.matrix_colonization[int(colonized)][self.cgen] = int(colonizer)
        self.cgen += 1

    def save(self):
        df = pd.DataFrame(self.matrix_colonization)
        file = np.array(self.matrix_colonization)
        fname = "phyloS" + str(self.seed) + '.csv'
        np.savetxt(fname, file, delimiter=',')
