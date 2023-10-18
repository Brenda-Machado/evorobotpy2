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
from evograd import expectation
from evograd.distributions import Normal

# Natural Evolution Strategies Algorithm


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

    def evaluate(self, candidate):
        
        self.policy.set_trainable_flat(candidate)
        self.policy.nn.normphase(
            0
        )  # normalization data is collected during the post-evaluation of the best sample of he previous generation
        eval_rews, eval_length = self.policy.rollout(
            self.policy.ntrials,
            seed= self.seed
        )
        self.steps += eval_length
        if eval_rews > self.bestestfit[0]:
            self.bestestfit = (eval_rews, candidate)

        self.fitness_eval.append(eval_rews)

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
        
        self.updateBest(self.bestestfit[0], self.bestestfit[1])
            
        return (eval_rews)    
    
    def pos_evaluate(self):
        
        self.avgfit = np.average(self.fitness_eval)  # compute the average fitness

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

    def run(self):

        self.setProcess()  # initialize class variables
        start_time = time.time()
        last_save_time = start_time
        elapsed = 0
        self.steps = 0
        self.bestestfit = (-99999999, None)
        self.fitness_eval = []
        print(
            "NES: seed %d maxmsteps %d batchSize %d stepsize %lf noiseStdDev %lf wdecay %d symseed %d nparams %d"
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

        while self.steps < self.maxsteps:

            self.fitnesses = []

            self.mu = torch.tensor(self.center, requires_grad=True)

            p = Normal(self.mu, self.noiseStdDev)

            pop = p.sample(n=self.batchSize)

            for i in pop:
                self.fitnesses.append(self.evaluate(i))

            self.pos_evaluate()

            self.fitnesses = torch.tensor(self.fitnesses , requires_grad=True)

            mean = expectation(self.fitnesses, pop, p=p)

            mean.backward()
            with torch.no_grad():
                self.mu += self.alpha * self.mu.grad
                self.mu.grad.zero_()
            self.center = self.mu.detach().numpy()
            
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
