#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/Brenda-Machado/evorobotpy2
   and has been written by Stefano Nolfi, stefano.nolfi@istc.cnr.it;
   Brenda S. Machado, brenda.silva.machado@grad.ufsc.br;
   and Arthur H. Bianchini, arthur.h.bianchini@grad.ufsc.br,
   requires es.py, policy.py, and evoalgo.py 
"""

import numpy as np
from numpy import zeros, dot, sqrt
import math
import time
from evoalgo import EvoAlgo
from utils import descendent_sort
import os
import configparser
import random


# SSSNE is a variant of SSS that uses niches and enviromental variation
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def loadhyperparameters(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.popsize = 20
            self.mutation = 0.02
            self.saveeach = 60
            self.number_niches = 25
            self.nGens = 50
            options = config.options("ALGO")
            for o in options:
                found = 0
                if o == "maxmsteps":
                    self.maxsteps = config.getint("ALGO", "maxmsteps") * 1000000
                    found = 1
                if o == "mutation":
                    self.mutation = config.getfloat("ALGO", "mutation")
                    found = 1
                if o == "popsize":
                    self.popsize = config.getint("ALGO", "popsize")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO", "saveeach")
                    found = 1
                if o == "number_niches":
                    self.number_niches = config.getint("ALGO", "number_niches")
                    found = 1
                if o == "nGens":
                    self.nGens = config.getint("ALGO", "nGens")
                    found = 1

                if found == 0:
                    print(
                        "\033[1mOption %s in section [ALGO] of %s file is unknown\033[0m"
                        % (o, filename)
                    )
                    print("available hyperparameters are: ")
                    print(
                        "maxmsteps [integer]       : max number of (million) steps (default 1)"
                    )
                    print("popsize [int]             : popsize (20)")
                    print("mutation [float]          : mutation (default 0.02)")
                    print(
                        "saveeach [integer]        : save file every N minutes (default 60)"
                    )

                    sys.exit()
        else:
            print(
                "\033[1mERROR: configuration file %s does not exist\033[0m"
                % (self.fileini)
            )

    def savedata(self, ceval, cgen, bfit, bgfit, avefit, aveweights):
        self.save()  #  save the best agent, the best postevaluated agent, and progress data across generations
        fname = self.filedir + "/S" + str(self.seed) + ".fit"
        fp = open(fname, "w")  # save summary
        fp.write(
            "Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f weightsize %.2f \n"
            % (
                self.seed,
                ceval / float(self.maxsteps) * 100,
                cgen,
                ceval / 1000000,
                self.bestfit,
                self.bestgfit,
                bfit,
                bgfit,
                avefit,
                aveweights,
            )
        )
        fp.close()
    
    def storePerformance(self):
        self.stat = np.append(
                self.stat,
                [
                    self.ceval,
                    self.bestfit,
                    self.bestgfit,
                    self.avgfit,
                ],
            )  # store performance across generations
    
    def evaluate(self, niche_flag=False):
        
        # Evaluate the population
        for i in range(self.popsize):
            self.policy.set_trainable_flat(self.pop[i])  # set policy parameters
            niche = i%self.number_niches
            if niche_flag:
                eval_rews, eval_length = self.policy.rollout(
                    self.policy.ntrials,
                    seed=(self.niches[niche]),
                )  # evaluate the individual
            else:
                # If normalize=1 we update the normalization vectors
                if self.policy.normalize == 1:
                    self.policy.nn.updateNormalizationVectors()

                self.env.seed(
                    self.policy.get_seed + self.cgen
                )  # set the environment seed, it changes every generation
                self.policy.nn.seed(
                    self.policy.get_seed + self.cgen
                )  # set the policy seed, it changes every generation

                eval_rews, eval_length = self.policy.rollout(
                    self.policy.ntrials
                )  # evaluate the individual
            self.fitness[i] = eval_rews  # store 
            self.ceval += eval_length  # Update the number of evaluations
            self.updateBest(
                self.fitness[i], self.pop[i]
            )  # Update data if the current offspring is better than current best
        # get best genotype
        self.other_fitness = self.fitness.copy()

        self.other_fitness, index = descendent_sort(
            self.other_fitness
        )  # create an index with the ID of the individuals sorted for
        bfit = self.other_fitness[index[0]]
        self.updateBest(
            bfit, self.pop[index[0]]
        )  # eventually update the genotype/ of the best individual so far

        # Postevaluate the best individual
        self.env.seed(
            self.policy.get_seed + 100000
        )  # set the environmental seed, always the same for the same seed
        self.policy.nn.seed(
            self.policy.get_seed + 100000
        )  # set the policy seed, always the same for the same seed
        self.policy.set_trainable_flat(
            self.pop[index[0]]
        )  # set the parameters of the policy
        eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)
        bgfit = eval_rews # best gen fit
        self.ceval += eval_length #current evaluation
        self.updateBestg(
            bgfit, self.pop[index[0]]
        )  # eventually update the genotype/ of the best post-evaluated individual

        # replace the worst half of the population with a mutated copy of the first half of the population
        half_popsize = int(self.popsize / 2)
        for i in range(half_popsize):
            self.pop[index[i + half_popsize]] = self.pop[index[i]] + (
                self.rg.randn(1, self.policy.nparams) * self.mutation
            )

        # display info
        """print(
            "Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f weightsize %.2f"
            % (
                self.seed,
                self.ceval / float(self.maxsteps) * 100,
                self.cgen,
                self.ceval / 1000000,
                self.bestfit,
                self.bestgfit,
                bfit,
                bgfit,
                np.average(self.fitness),
                np.average(np.absolute(self.pop[index[0]])),
            )
        )"""

    def inniche(self, niche): 
        bestg = None
        bfitness = -999999
        for o in range(niche, self.popsize, self.number_niches):
            if self.fitness[o] >= bfitness:
                bestg = o
                bfitness = self.fitness[o]

        # mutate offspring
        aux = np.array(self.pop[bestg] + self.rg.randn(1, self.policy.nparams) * self.mutation)
        offspring = aux
        self.policy.set_trainable_flat(offspring[0])  # set policy parameters
        eval_rews, eval_length = self.policy.rollout(
            self.policy.ntrials,
            seed=(self.niches[bestg%self.number_niches]),
        )  # evaluate the individual
        self.ceval += eval_length  # Update the number of evaluations

        if eval_rews >= self.fitness[bestg]:
            self.pop[bestg] = offspring[0]
            self.fitness[bestg] = eval_rews

    def intraniche(self, out=True):
        for niche in range(self.number_niches):
            self.inniche(niche)   # evaluate samples
        self.cgen += 1
        

    def interniche(self): 
        fitMatrix = np.zeros(shape=(self.number_niches, self.number_niches))
        self.current_colonized = [i for i in range(self.number_niches)]
        self.boffspring = [0 for _ in range(self.number_niches)]
        self.temp_pop = self.pop[:]
        hasColonized = False
        for niche in range(self.number_niches):
            bfitness = -999999
            for o in range(niche, self.popsize, self.number_niches):
                if self.fitness[o] > bfitness:
                    self.boffspring[niche] = o
                    bfitness = self.fitness[o]
            for miche in range(self.number_niches):
                if miche != niche:
                    # Evaluate center of niche n in niche m
                    self.policy.set_trainable_flat(self.pop[self.boffspring[niche]])  # set policy parameters
                    eval_rews, eval_length = self.policy.rollout(
                        self.policy.ntrials,
                        seed=(self.niches[miche]),
                    )  # evaluate the individual
                    fitMatrix[niche][miche] = eval_rews
                else:
                    fitMatrix[niche][miche] = -99999999


        for miche in range(self.number_niches):
            biche = np.argmax(fitMatrix[:][miche])
            maxFit = fitMatrix[biche][miche]
            if maxFit > self.fitness[miche]:
                biche = np.argmax(fitMatrix[:][miche])
                #print("Niche", biche+1, "colonized niche", miche+1)
                hasColonized = True
                for i in range(self.number_niches):
                    fitMatrix[i][biche] = -99999999
                    fitMatrix[miche][i] = -99999999
                # Replace i with o in niche m
                self.fitness[miche] = maxFit
                self.temp_pop[self.boffspring[miche]] = self.pop[self.boffspring[biche]]
        if hasColonized:
            self.pop = self.temp_pop[:]


    def run(self):

        self.loadhyperparameters()  # initialize hyperparameters
        start_time = time.time()  # start time
        nparams = self.policy.nparams  # number of parameters
        self.ceval = 0  # current evaluation
        self.cgen = 0  # current generation
        self.rg = np.random.RandomState(
            self.seed
        )  # create a random generator and initialize the seed
        self.pop = self.rg.randn(self.popsize, nparams)  # population
        self.niches = [[random.randint(1, self.number_niches*10**5) for _ in range(self.policy.ntrials)] for _ in range(self.number_niches)]
        self.fitness = zeros(self.popsize)  # fitness
        self.stat = np.arange(
            0, dtype=np.float64
        )  # initialize vector containing performance across generations

        assert (self.popsize % 2) == 0, print(
            "the size of the population should be odd"
        )

        # initialze the population
        for i in range(self.popsize):
            self.pop[i] = self.policy.get_trainable_flat()

        """print(
            "SSSNE: seed %d maxmsteps %d popSize %d noiseStdDev %lf nparams %d"
            % (self.seed, self.maxsteps / 1000000, self.popsize, self.mutation, nparams)
        )"""

        # main loop
        elapsed = 0

        self.evaluate(niche_flag=True)

        while self.ceval < self.maxsteps:

            for gen in range(self.nGens):
                self.evaluate()
                self.storePerformance()
                self.intraniche()

            self.interniche()

            # If normalize=1 we update the normalization vectors
            if self.policy.normalize == 1:
                self.policy.nn.updateNormalizationVectors()

            # save data
            if (time.time() - self.last_save_time) > (self.saveeach * 60):
                self.save()
                self.last_save_time = time.time()

        self.savedata(
            self.ceval,
            self.cgen,
            self.bestfit,
            self.bestgfit,
            np.average(self.fitness),
            np.average(np.absolute(self.pop[0])),
        )
        end_time = time.time()
        #print("Simulation time: %dm%ds " % (divmod(end_time - start_time, 60)))
        print(1000 - self.bestgfit)
