#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/Brenda-Machado/evorobotpy2
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it,
   Brenda S. Machado, brenda.silva.machado@grad.ufsc.br, and Arthur H. Bianchini, arthur.h.bianchini@grad.ufsc.br,
   salimans.py include an implementation of the OpenAI-ES algorithm described in
   Salimans T., Ho J., Chen X., Sidor S & Sutskever I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. arXiv:1703.03864v2
   requires es.py, policy.py, and evoalgo.py; and modified to work with niches and environmental differentiation
"""

import numpy as np
from numpy import zeros, ones, dot, sqrt
import math
import time
from evoalgo import EvoAlgo
from utils import ascendent_sort
import sys
import os
import configparser
import random

# Parallel implementation of Open-AI-ES algorithm developed by Salimans et al. (2017)
# the workers evaluate a fraction of the population in parallel
# the master post-evaluate the best sample of the last generation and eventually update the input normalization vector
# niches and environmental differentiation are supported


class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def loadhyperparameters(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.stepsize = 0.01
            self.batchSize = 20
            self.noiseStdDev = 0.02
            self.wdecay = 0
            self.symseed = 1
            self.saveeach = 60
            self.number_niches = 5
            options = config.options("ALGO")
            for o in options:
                found = 0
                if o == "maxmsteps":
                    self.maxsteps = config.getint("ALGO", "maxmsteps") * 1000000
                    found = 1
                if o == "stepsize":
                    self.stepsize = config.getfloat("ALGO", "stepsize")
                    found = 1
                if o == "noisestddev":
                    self.noiseStdDev = config.getfloat("ALGO", "noiseStdDev")
                    found = 1
                if o == "samplesize":
                    self.batchSize = config.getint("ALGO", "sampleSize")
                    found = 1
                if o == "wdecay":
                    self.wdecay = config.getint("ALGO", "wdecay")
                    found = 1
                if o == "symseed":
                    self.symseed = config.getint("ALGO", "symseed")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO", "saveeach")
                    found = 1
                if o == "number_niches":
                    self.number_niches = config.getint("ALGO", "number_niches")
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
                    print(
                        "stepsize [float]          : learning stepsize (default 0.01)"
                    )
                    print("samplesize [int]          : popsize/2 (default 20)")
                    print("noiseStdDev [float]       : samples noise (default 0.02)")
                    print(
                        "wdecay [0/2]              : weight decay (default 0), 1 = L1, 2 = L2"
                    )
                    print(
                        "symseed [0/1]             : same environmental seed to evaluate symmetrical samples [default 1]"
                    )
                    print(
                        "saveeach [integer]        : save file every N minutes (default 60)"
                    )

                    sys.exit()
        else:
            print(
                "\033[1mERROR: configuration file %s does not exist\033[0m"
                % (self.fileini)
            )

    def setProcess(self):
        self.loadhyperparameters()  # load hyperparameters
        self.avecenter = None
        self.nGens = 50             # number of generations until migration occurs
        self.bniche = None    
        self.colonizer = [np.nan for _ in range(self.number_niches)]
        self.fitness = np.zeros(self.number_niches)
        self.avecenters = np.zeros(self.number_niches)
        self.centers = [np.copy(self.policy.get_trainable_flat()) for i in range(self.number_niches)]  # the initial centroids
        self.nparams = len(self.centers[0])  # number of adaptive parameters
        self.cgen = 1  # currrent generation
        self.samplefitness = zeros(self.batchSize * 2)  # the fitness of the samples
        self.samples = None  # the random samples
        self.m = zeros(self.nparams)  # Adam: momentum vector
        self.v = zeros(self.nparams)  # Adam: second momentum vector (adam)
        self.epsilon = 1e-08  # Adam: To avoid numerical issues with division by zero...
        self.beta1 = 0.9  # Adam: beta1
        self.beta2 = 0.999  # Adam: beta2
        self.bestgfit = -99999999  # the best generalization fitness
        self.bfit = 0  # the fitness of the best sample
        self.gfit = (
            0  # the postevaluation fitness of the best sample of last generation
        )
        self.rs = None  # random number generator
        self.inormepisodes = (
            self.batchSize * 2 * self.policy.ntrials / 100.0
        )  # number of normalization episode for generation (1% of generation episodes)
        self.tnormepisodes = (
            0.0  # total epsidoes in which normalization data should be collected so far
        )
        self.normepisodes = 0  # numer of episodes in which normalization data has been actually collected so far
        self.normalizationdatacollected = (
            False  # whether we collected data for updating the normalization vector
        )
        for niche in range(self.numberNiches):
            self.phyloy[niche+self.numberNiches*niche][0] = niche+1
            self.phyloy[niche+self.numberNiches*niche][1] = niche+1

    def savedata(self):
        self.save(esne=True)  # save the best agent so far, the best postevaluated agent so far, and progress data across generations
        fname = self.filedir + "/S" + str(self.seed) + ".fit"
        fp = open(fname, "w")  # save summary
        fp.write(
            "Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f bestniche %d avgfit %.2f paramsize %.2f \n"
            % (
                self.seed,
                self.steps / float(self.maxsteps) * 100,
                self.cgen,
                self.steps / 1000000,
                self.bestfit,
                self.bestgfit,
                self.bniche,
                self.avgfit,
                self.avecenter,
            )
        )
        fp.close()

    def evaluate(self, niche, oniche=None):
        oniche_flag = 1
         
        if oniche is None:
            oniche = niche
            oniche_flag = 0
        
        cseed = (
            self.seed + self.cgen * self.batchSize
        )  # Set the seed for current generation (master and workers have the same seed)
        self.rs = np.random.RandomState(cseed)
        self.samples = self.rs.randn(self.batchSize, self.nparams)

        # evaluate samples
        candidate = np.arange(self.nparams, dtype=np.float64)
        for b in range(self.batchSize):
            for bb in range(2):
                if bb == 0:
                    candidate = self.centers[niche] + self.samples[b, :] * self.noiseStdDev
                else:
                    candidate = self.centers[niche] - self.samples[b, :] * self.noiseStdDev
                self.policy.set_trainable_flat(candidate)
                self.policy.nn.normphase(
                    0
                )  # normalization data is collected during the post-evaluation of the best sample of he previous generation
                eval_rews, eval_length = self.policy.rollout(
                    self.policy.ntrials,
                    seed=self.niches[oniche],
                )
                self.samplefitness[b * 2 + bb] = eval_rews
                self.steps += eval_length

        fitness, self.index = ascendent_sort(self.samplefitness)  # sort the fitness
        self.avgfit = np.average(fitness)  # compute the average fitness
        if oniche_flag:
            self.fitness[oniche] = self.avgfit

        self.bfit = fitness[(self.batchSize * 2) - 1]
        bidx = self.index[(self.batchSize * 2) - 1]
        if (bidx % 2) == 0:  # regenerate the genotype of the best samples
            bestid = int(bidx / 2)
            self.bestsol = self.centers[niche] + self.samples[bestid] * self.noiseStdDev
        else:
            bestid = int(bidx / 2)
            self.bestsol = self.centers[niche] - self.samples[bestid] * self.noiseStdDev

        self.updateBest(
            self.bfit, self.bestsol
        )  # Stored if it is the best obtained so far

        # postevaluate best sample of the last generation
        # in openaiesp.py this is done the next generation, move this section before the section "evaluate samples" to produce identical results
        gfit = 0
        if self.bestsol is not None:
            self.policy.set_trainable_flat(self.bestsol)
            self.tnormepisodes += self.inormepisodes
            for t in range(self.policy.nttrials):
                if (
                    self.policy.normalize == 1
                    and self.normepisodes < self.tnormepisodes
                ):
                    self.policy.nn.normphase(1)
                    self.normepisodes += 1  # we collect normalization data
                    self.normalizationdatacollected = True
                else:
                    self.policy.nn.normphase(0)
                eval_rews, eval_length = self.policy.rollout(
                    1, seed=(self.seed + 100000 + t)
                )
                gfit += eval_rews
                self.steps += eval_length
            gfit /= self.policy.nttrials
            self.updateBestg(gfit, self.bestsol)
        
    def optimize(self, niche):
        popsize = self.batchSize * 2  # compute a vector of utilities [-0.5,0.5]
        utilities = zeros(popsize)
        for i in range(popsize):
            utilities[self.index[i]] = i
        utilities /= popsize - 1
        utilities -= 0.5

        weights = zeros(
            self.batchSize
        )  # Assign the weights (utility) to samples on the basis of their fitness rank
        for i in range(self.batchSize):
            idx = 2 * i
            weights[i] = (
                utilities[idx] - utilities[idx + 1]
            )  # merge the utility of symmetric samples

        g = 0.0
        i = 0
        while (
            i < self.batchSize
        ):  # Compute the gradient (the dot product of the samples for their utilities)
            gsize = -1
            if (
                self.batchSize - i < 500
            ):  # if the popsize is larger than 500, compute the gradient for multiple sub-populations
                gsize = self.batchSize - i
            else:
                gsize = 500
            g += dot(weights[i : i + gsize], self.samples[i : i + gsize, :])
            i += gsize
        g /= popsize  # normalize the gradient for the popsize

        if self.wdecay == 1:
            globalg = -g + 0.005 * self.centers[niche]  # apply weight decay
        else:
            globalg = -g

        # adam stochastic optimizer
        a = (
            self.stepsize
            * sqrt(1.0 - self.beta2 ** self.cgen)
            / (1.0 - self.beta1 ** self.cgen)
        )
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (globalg * globalg)
        dCenter = -a * self.m / (sqrt(self.v) + self.epsilon)

        self.centers[niche] += (
            dCenter  # move the center in the direction of the momentum vectors
        )
        self.avecenters[niche] = np.average(np.absolute(self.centers[niche]))
        
    def storePerformance(self):
        self.stat = np.append(
                self.stat,
                [
                    self.steps,
                    self.bestfit,
                    self.bestgfit,
                    self.bniche,
                    self.avgfit,
                    self.avecenter,
                ],
            )  # store performance across generations

    def intraniche(self):

        for niche in range(self.number_niches):
            self.evaluate(niche)   # evaluate samples
            self.optimize(niche)   # estimate the gradient and move the centroid in the gradient direction
        print(
                "Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f"
                % (
                    self.seed,
                    self.steps / float(self.maxsteps) * 100,
                    self.cgen,
                    self.steps / 1000000,
                    self.bestfit,
                    self.bestgfit,
                )
            )
        self.cgen += 1
        self.storePerformance()

    def interniche(self): 
        self.colonized = [False for _ in range(self.number_niches**2)]
        fitMatrix = np.zeros(shape=(self.number_niches, self.number_niches))

        for niche in range(self.number_niches):
            for miche in range(self.number_niches):
                if miche != niche:
                    # Evaluate center of niche n in niche m
                    self.evaluate(niche, miche)
                    fitMatrix[niche][miche] = self.avgfit
                else:
                    fitMatrix[niche][miche] = -99999999
                    
        for miche in range(self.number_niches):
            # biche = best niche in miche
            biche = np.argmax(fitMatrix[:][miche])
            maxFit = fitMatrix[biche][miche]
            if maxFit > self.fitness[miche]:
                biche = np.argmax(fitMatrix[:][miche])
                print("Niche", biche+1, "colonized niche", miche+1)
                self.colonizer[miche] = biche

                for i in range(self.number_niches):
                    fitMatrix[i][biche] = -99999999
                    fitMatrix[miche][i] = -99999999

                # Replace i with o in niche m
                self.fitness[miche] = maxFit
                # Replace center of niche m with center of niche j
                self.centers[miche] = self.centers[biche]

    def phylogenetic(self, biche, miche):
        # if the best niche was already colonized, then its colonizer will be represented instead
        if np.isnan(self.colonizer[biche]):
            colonizer = biche
        else:
            colonizer = self.colonizer[biche]
        colonized_old_colonizer = self.colonizer[miche]
        new_colonization_graph = miche+self.numberNiches*colonizer
        cgen = self.cgen//self.nGens
        self.phyloy[new_colonization_graph][cgen] = colonizer+1
        self.phyloy[new_colonization_graph][cgen+1] = miche+1
        self.phyloy[colonized_old_colonizer][cgen+1] = miche+1
        for graph in range(0, self.numberNiches**2, miche):
            self.colonized[miche] = True

    
    def phylogenetic_end(self):
        cgen = self.cgen//self.nGens
        for graph in range(self.numberNiches**2):
            if not self.colonized[graph]:
                self.phyloy[graph][cgen+1] = self.phyloy[graph][cgen]

    def run(self):

        self.setProcess()  # initialize class variables
        start_time = time.time()
        last_save_time = start_time
        elapsed = 0
        self.steps = 0
        print(
            "Salimans: seed %d maxmsteps %d batchSize %d stepsize %lf noiseStdDev %lf wdecay %d symseed %d nparams %d"
            % (
                self.seed,
                self.maxsteps / 1000000,
                self.batchSize,
                self.stepsize,
                self.noiseStdDev,
                self.wdecay,
                self.symseed,
                self.nparams,
            )
        )

        random_niches = []
        num_random_niches = self.number_niches*100

        for _ in range(num_random_niches):
            random_niches.append([random.randint(1, num_random_niches*10) for _ in range(self.policy.ntrials)])

        self.niches = [0 for _ in range(self.number_niches)]
 
        for niche in range(self.number_niches): 
            self.niches[niche] = random_niches[niche]

        remove_first_gen = 1

        while self.steps < self.maxsteps:

            for _ in range(self.nGens):
                self.intraniche()
                
            if remove_first_gen:
                self.cgen -= 1
                remove_first_gen = 0
            
            self.bniche = np.argmax(self.fitness)
            
            print("Intraniche finished")
            
            self.interniche()

            print("Interniche finished")

            self.avgfit = np.average(self.fitness)
            
            self.avecenter = np.average(self.avecenters)   

            if (time.time() - last_save_time) > (self.saveeach * 60):
                self.savedata()  # save data on files
                last_save_time = time.time()

            if self.normalizationdatacollected:
                self.policy.nn.updateNormalizationVectors()  # update the normalization vectors with the new data collected
                self.normalizationdatacollected = False

            print(
                "Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f bestniche %d avg %.2f weightsize %.2f"
                % (
                    self.seed,
                    self.steps / float(self.maxsteps) * 100,
                    self.cgen,
                    self.steps / 1000000,
                    self.bestfit,
                    self.bestgfit,
                    self.bniche,
                    self.avgfit,
                    self.avecenter,
                )
            )

        self.savedata()

        # print simulation time
        end_time = time.time()
        print("Simulation time: %dm%ds " % (divmod(end_time - start_time, 60)))