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
from numpy import zeros, ones, dot, sqrt

class Phylogenetic:
    def __init__(self, number_niches, max_gens, seed):
        self.matrix_colonization = np.nan(shape=(max_gens, number_niches))
        self.seed = seed
        self.cgen = 0

    def initial_state(self):
        for i in range(len(self.matrix_colonization)):
            self.matrix_colonization[0][i] = i

    def colonize(self, colonizer, colonized):
        self.matrix_colonization[self.cgen][colonized] = colonizer
        self.cgen += 1

    def save(self):
        file = np.array(self.matrix_colonization)
        fname = "phyloS" + str(self.seed) + '.csv'
        np.savetxt(fname, file, delimiter=',')
