#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it

   evoalgo.py contains methods for showing, saving and loading data 

 """

import random
import numpy as np
import time
from data_interfaces.conditions.initial import InitialConditions
from data_interfaces.stats.run import RunStats
from curriculum_learning.curriculum.base_grid import generate_grid
from data_interfaces.utils import set_root
set_root('evorobot-baseline')

ENVIRONMENT_FEATURES = dict(
    xdpole=6,
    xbipedal=200
)

class EvoAlgo(object):
    def __init__(self, env, policy, seed, fileini, filedir):
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

        self.cgen = None
        self.test_limit_stop = None

        env_features = self._get_env_features()
        self.initialize_data_managers(env_features=env_features)

    def _get_env_features(self):
        n_features = ENVIRONMENT_FEATURES[self.__env_name]
        features = [f"x{i}" for i in range(n_features)] + ["performance"]
        return features

    def initialize_data_managers(self, env_features=[], stats_features=[]):
        self.base_grid = generate_grid(5)
        upload_reference = 'baseline'

        self.initialconditions = InitialConditions(
            self.__env_name,
            self.seed,
            env_features,
            trials=self.policy_trials,
            upload_reference=upload_reference
        )

        self.runstats = RunStats(
            self.__env_name,
            self.seed,
            stats_features,
            upload_reference=upload_reference
        )

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

    def process_initial_conditions(self):
        self.generation_conditions = self.evaluate_center(self.policy.ntrials, self.evaluation_seed)
        self.initialconditions.save_stg(self.generation_conditions, self.cgen)

    def evaluate_center(self, ntrials=10, seed=None, curriculum=None):
        seed = seed if seed else self.cgen
        candidate = self.center
        self.policy.set_trainable_flat(candidate)
        self.policy.nn.normphase(0)
        self.policy.rollout(ntrials, seed=seed, progress=self.progress, save_env=True)
        return self.policy.rollout_env

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
        self.initialconditions.save()
        self.runstats.save()

    def process_conditions(self):
        self.process_initial_conditions()

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
        eval_rews, eval_length = self.policy.rollout(ntrials, progress=0, seed=self.policy.get_seed + 100000)
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
