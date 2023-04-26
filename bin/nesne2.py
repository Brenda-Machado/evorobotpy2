#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/Brenda-Machado/evorobotpy2
   and has been written by Brenda Machado.
   It is an implementation of the NES algorithm described in
   Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, and Jürgen Schmidhuber.
   2011. Natural Evolution Strategies. https://doi.org/10.48550/ARXIV.1106.4487

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
import torch
import random
from evograd import expectation
from evograd.distributions import Normal

# Natural Evolution Strategies Niche Evolution

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
        self.center = np.copy(self.policy.get_trainable_flat())  # the initial centroid
        self.nparams = len(self.center)  # number of adaptive parameters
        self.cgen = 0  # currrent generation
        self.samplefitness = zeros(self.batchSize * 2)  # the fitness of the samples
        self.samples = None  # the random samples
        self.m = zeros(self.nparams)  # Adam: momentum vector
        self.v = zeros(self.nparams)  # Adam: second momentum vector (adam)
        self.epsilon = 1e-08  # Adam: To avoid numerical issues with division by zero...
        self.alpha = 0.2 # NES: learning rate
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
        self.number_niches = 25
        self.fitness = [-9999 for _ in range(self.number_niches)]
        self.colonizer = [np.nan for _ in range(self.number_niches)]
        self.candidates = [0 for _ in range(self.number_niches)]
        self.centers = [np.copy(self.policy.get_trainable_flat()) for i in range(self.number_niches)]  # the initial centroids
        

    def savedata(self):
        self.save()  # save the best agent so far, the best postevaluated agent so far, and progress data across generations
        fname = self.filedir + "/S" + str(self.seed) + ".fit"
        fp = open(fname, "w")  # save summary
        fp.write(
            "Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f bestsam %.2f avgfit %.2f\n"
            % (
                self.seed,
                self.steps / float(self.maxsteps) * 100,
                self.cgen,
                self.steps / 1000000,
                self.bestfit,
                self.bestgfit,
                self.bfit,
                self.avgfit,
            )
        )
        fp.close()

    def evaluate(self, candidate, oniche=None):
        oniche_flag = True
        if self.niche == self.number_niches:
            self.cgen += 1
            if self.cgen % 25 == 0:
                self.interniche()
        
        if oniche is None:        
            oniche_flag = False

        if not oniche_flag:
                self.niche = self.niche % self.number_niches
                oniche = self.niche
        
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
                    candidate = self.centers[oniche] + self.samples[b, :] * self.noiseStdDev
                else:
                    candidate = self.centers[oniche] - self.samples[b, :] * self.noiseStdDev
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
        

        self.bfit = fitness[(self.batchSize * 2) - 1]
        bidx = self.index[(self.batchSize * 2) - 1]
        if (bidx % 2) == 0:  # regenerate the genotype of the best samples
            bestid = int(bidx / 2)
            self.bestsol = self.centers[oniche] + self.samples[bestid] * self.noiseStdDev
        else:
            bestid = int(bidx / 2)
            self.bestsol = self.centers[oniche] - self.samples[bestid] * self.noiseStdDev

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
            if gfit > self.fitness[oniche] and not oniche_flag:
                self.fitness[oniche] = gfit
        return gfit

    def interniche(self): 
        self.colonized = [False for _ in range(self.number_niches**2)]
        fitMatrix = np.zeros(shape=(self.number_niches, self.number_niches))

        for niche in range(self.number_niches):
            for miche in range(self.number_niches):
                if miche != niche:
                    # Evaluate center of niche n in niche m
                    fitMatrix[niche][miche] = self.eval(self.candidates[niche], miche)
                else:
                    fitMatrix[niche][miche] = -99999999

        for miche in range(self.number_niches):
            # biche = best niche in miche
            biche = np.argmax([fitMatrix[j][miche] for j in range(self.number_niches)])
            maxFit = fitMatrix[biche][miche]

            
            if maxFit > self.fitness[miche]:
                print("Niche", biche+1, "colonized niche", miche+1)
                self.colonizer[miche] = biche

                for i in range(self.number_niches):
                    fitMatrix[biche][i] = -99999999

                # Replace i with o in niche m
                self.fitness[miche] = maxFit
                # Replace center of niche m with center of niche j
                self.candidates[miche] = self.candidates[biche]

    def run(self):

        self.setProcess()  # initialize class variables
        start_time = time.time()
        last_save_time = start_time
        elapsed = 0
        self.steps = 0
        self.bestestfit = (-99999999, None)
        self.fitness_eval = []
        print(
            "NES-NE: seed %d maxmsteps %d batchSize %d stepsize %lf noiseStdDev %lf wdecay %d symseed %d nparams %d"
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

        self.niche = 0

        while self.steps < self.maxsteps:

            self.fitnesses = []

            self.mu = torch.tensor(self.center, requires_grad=True)

            p = Normal(self.mu, self.noiseStdDev)

            pop = p.sample(n=self.batchSize)

            for i in pop:
                self.fitnesses.append(self.evaluate(i))

            self.fitnesses = torch.tensor(self.fitnesses , requires_grad=True)

            mean = expectation(self.fitnesses, pop, p=p)

            mean.backward()
            with torch.no_grad():
                self.mu += self.alpha * self.mu.grad
                self.mu.grad.zero_()
            
            self.stat = np.append(
                self.stat,
                [
                    self.steps,
                    self.bestfit,
                    self.bestgfit,
                    self.bfit,
                    self.avgfit,
                ],
            )  # store performance across generations

            if (time.time() - last_save_time) > (self.saveeach * 60):
                self.savedata()  # save data on files
                last_save_time = time.time()

            if self.normalizationdatacollected:
                self.policy.nn.updateNormalizationVectors()  # update the normalization vectors with the new data collected
                self.normalizationdatacollected = False

            print(
                "Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f bestsam %.2f avg %.2f"
                % (
                    self.seed,
                    self.steps / float(self.maxsteps) * 100,
                    self.cgen,
                    self.steps / 1000000,
                    self.bestfit,
                    self.bestgfit,
                    self.bfit,
                    self.avgfit,
                )
            )

        self.savedata()  # save data at the end of evolution

        # print simulation time
        end_time = time.time()
        print("Simulation time: %dm%ds " % (divmod(end_time - start_time, 60)))
