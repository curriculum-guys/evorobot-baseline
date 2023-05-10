#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it

   evoalgo.py contains methods for showing, saving and loading data 

 """

import random
import math
import numpy as np
import time
from data_interfaces.conditions.initial import InitialConditions
from data_interfaces.stats.run import RunStats
from curriculum_learning.curriculum.base_grid import generate_grid
from data_interfaces.conditions.base import BaseConditions
from data_interfaces.utils import set_root
set_root('evorobot-baseline')

class EvoAlgo(object):
    def __init__(self, env, policy, seed, fileini, filedir, icfeatures=[], statsfeatures=[]):
        self.env = env                       # the environment
        self.policy = policy                 # the policy
        self.seed = seed                     # the seed of the experiment
        self.fileini = fileini               # the name of the file with the hyperparameters
        self.filedir = filedir               # the directory used to save/load files
        self.bestfit = -999999999.0          # the fitness of the best agent so far
        self.bestsol = None                  # the genotype of the best agent so far
        self.bestgfit = -999999999.0         # the performance of the best post-evaluated agent so far
        self.bestgsol = None                 # the genotype of the best postevaluated agent so far
        self.stat = np.arange(0, dtype=np.float64) # a vector containing progress data across generations
        self.avgfit = 0.0                    # the average fitness of the population
        self.last_save_time = time.time()    # the last time in which data have been saved
        self.policy_trials = self.policy.ntrials

        upload_reference = 'baseline'

        self.initialconditions = InitialConditions(
            self.__env_name,
            seed,
            icfeatures,
            trials=self.policy_trials,
            upload_reference=upload_reference
        )

        self.runstats = RunStats(
            self.__env_name,
            seed,
            statsfeatures,
            upload_reference=upload_reference
        )

        self.grid_conditions = generate_grid(5)
        self.batch_size = len(self.grid_conditions) * 5
        self.grid_batch = []
        self.curriculum = None

        self.cgen = None
        self.test_limit_stop = None
    
    @property
    def __env_name(self):
        return self.fileini.split('/')[2].split('/')[0]

    @property
    def progress(self):
        return self.steps / float(self.maxsteps) * 100

    @property
    def cgen(self):
        return self._cgen

    @cgen.setter
    def cgen(self, cgen):
        self._cgen = cgen

    @property
    def evaluation_seed(self):
        return self.seed + (self.cgen * self.batchSize)

    def generate_conditions(self, n_conditions, random_conditions=False):
        r = random.randint(1, n_conditions) if random_conditions else 1
        return [self.reset_env(i * r) for i in range(n_conditions)]

    def process_grid_conditions(self):
        conditions = self.evaluate_center(
            ntrials=len(self.grid_conditions),
            seed=self.evaluation_seed,
            curriculum=self.grid_conditions
        )
        if self.progress > 10:
            self.grid_batch += conditions
            self.grid_batch = self.grid_batch[-self.batch_size:]
        return conditions

    def process_initial_conditions(self):
        self.generation_conditions = self.evaluate_center(self.policy.ntrials, self.evaluation_seed)
        self.initialconditions.save_stg(self.generation_conditions, self.cgen)

    def evaluate_center(self, ntrials=10, seed=None, curriculum=None):
        seed = seed if seed else self.cgen
        candidate = self.center
        self.policy.set_trainable_flat(candidate)
        self.policy.nn.normphase(0)
        self.policy.rollout(ntrials, seed=seed, curriculum=curriculum, save_env=True)
        return self.policy.rollout_env

    def curriculum_subsets(self):
        e = len(self.grid_conditions)
        p = lambda i: np.mean([self.grid_batch[i + e*j] for j in range(5)])
        pg = [p(i) for i in range(e)]
        n = self.policy_trials
        f = lambda x: x ** 3
        norm = lambda p: (p - min(pg))/(max(pg) - min(pg))

        subsets = []
        for i in range(n):
            subset = []
            for j in range(e):
                ps = norm(p(j))
                if ps > f(i/n) and ps <= f(i+1/n): 
                    subset.append(self.grid_conditions[j])
            subsets.append(subset)
        return subsets

    def generate_curriculum(self):
        if self.progress > 10 and len(self.grid_batch) == self.batch_size:
            subsets = self.curriculum_subsets()

            curriculum = []
            for s in range(self.policy_trials):
                select_subset = lambda i: subsets[i] if len(subsets[i]) > 0 else select_subset(i+1)
                subset = select_subset(s)
                i = random.randint(0, len(subset)-1)
                curriculum.append(subset[i])
            self.curriculum = curriculum

    def save_summary(self):
        data = [
            '%d' % (self.steps / 1000000),
            '%.2f' % self.bestfit,
            '%.2f' % self.bestgfit,
            '%.2f' % self.bfit,
            '%.2f' % self.avgfit,
            '%.2f' % self.avecenter
        ]
        self.runstats.save_stg(data, self.cgen)

    def save_all(self):
        self.runstats.save()
        self.initialconditions.save()

    def process_conditions(self):
        self.process_initial_conditions()
        self.process_grid_conditions()

    def reset(self):
        self.bestfit = -999999999.0
        self.bestsol = None
        self.bestgfit = -999999999.0
        self.bestgsol = None
        self.stat = np.arange(0, dtype=np.float64)
        self.avgfit = 0.0
        self.last_save_time = time.time()

    def reset_env(self, salt):
        # Reset Env method depends on the algorithm
        raise NotImplementedError

    def run(self, nevals):
        # Run method depends on the algorithm
        raise NotImplementedError

    def test(self, testfile):  # postevaluate an agent 
        if (self.policy.test == 1 and "Bullet" in self.policy.environment):
            self.env.render(mode="human")    # Pybullet render require this initialization
        if testfile is not None:
            if self.filedir.endswith("/"):
                fname = self.filedir + testfile
            else:
                fname = self.filedir + "/" + testfile
            if (self.policy.normalize == 0):
                bestgeno = np.load(fname)
            else:
                geno = np.load(fname)
                for i in range(self.policy.ninputs * 2):
                    self.policy.normvector[i] = geno[self.policy.nparams + i]
                bestgeno = geno[0:self.policy.nparams]
                self.policy.nn.setNormalizationVectors()
            self.policy.set_trainable_flat(bestgeno)
        else:
            self.policy.reset()
        if (self.policy.nttrials > 0):
            ntrials = self.policy.nttrials
        else:
            ntrials = self.policy.ntrials
        eval_rews, eval_length = self.policy.rollout(ntrials, seed=self.policy.get_seed + 100000)
        print("Postevauation: Average Fitness %.2f Total Steps %d" % (eval_rews, eval_length))
        self.save_test_stats(eval_rews, eval_length)

    def updateBest(self, fit, ind):  # checks whether this is the best agent so far and in case store it
        if fit > self.bestfit:
            self.bestfit = fit
            if (self.policy.normalize == 0):
                self.bestsol = np.copy(ind)
            else:
                self.bestsol = np.append(ind,self.policy.normvector)

    def updateBestg(self, fit, ind): # checks whether this is the best postevaluated agent so far and eventually store it
        if fit > self.bestgfit:
            self.bestgfit = fit
            if (self.policy.normalize == 0):
                self.bestgsol = np.copy(ind)
            else:
                self.bestgsol = np.append(ind,self.policy.normvector)

    def save_best_stats(self):        # save the best agent so far, the best postevaluated agent so far
        self.runstats.save_metric(self.bestsol, 'bestsol')
        self.runstats.save_metric(self.bestgsol, 'bestgsol')

    def save_test_stats(self, avg, steps):
        self.runstats.save_test(avg, steps)
