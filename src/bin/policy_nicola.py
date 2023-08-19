#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it

   policy.py takes care of the creation and evaluation of the policy
   Requires the net.so library that can be obtained by compiling with cython the following files contained in the ./lib directory:
   evonet.cpp, evonet.h, utilities.cpp, utilities.h, net.pxd, net.pyx and setupevonet.py
   with the commands: cd ./evorobotpy/lib; python3 setupevonet.py build_ext –inplace; cp net*.so ../bin
   Also requires renderWorld.py to display neurons and to display the behavior of Er environments

"""
import net
import numpy as np
import configparser
import time
import itertools

def create_states():
    # init list of possible starting positions
    stateRanges_0 = 1.944
    stateRanges_1 = 1.215
    stateRanges_2 = 0.10472
    stateRanges_3 = 0.135088
    stateRanges_4 = 0.10472
    stateRanges_5 = 0.135088
    states_0 = np.linspace(-stateRanges_0, stateRanges_0, 5)
    states_1 = np.linspace(-stateRanges_1, stateRanges_1, 5)
    states_2 = np.linspace(-stateRanges_2, stateRanges_2, 5)
    states_3 = np.linspace(-stateRanges_3, stateRanges_3, 5)
    states_4 = np.linspace(-stateRanges_4, stateRanges_4, 5)
    states_5 = np.linspace(-stateRanges_5, stateRanges_5, 5)
    states = [states_0, states_1, states_2, states_3, states_4, states_5]

    return  states

def create_test_states():
    # init list of possible starting positions
    stateRanges_0 = 1.944
    stateRanges_1 = 1.215
    stateRanges_2 = 0.10472
    stateRanges_3 = 0.135088
    stateRanges_4 = 0.10472
    stateRanges_5 = 0.135088
    states_0 = np.linspace(-stateRanges_0, stateRanges_0, 3)
    states_1 = np.linspace(-stateRanges_1, stateRanges_1, 3)
    states_2 = np.linspace(-stateRanges_2, stateRanges_2, 3)
    states_3 = np.linspace(-stateRanges_3, stateRanges_3, 3)
    states_4 = np.linspace(-stateRanges_4, stateRanges_4, 3)
    states_5 = np.linspace(-stateRanges_5, stateRanges_5, 3)
    states = [states_0, states_1, states_2, states_3, states_4, states_5]
    return  states

class Policy(object):
    def __init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test):
        # Copy environment
        self.env = env
        self.seed = seed
        self.rs = np.random.RandomState(seed)
        self.ninputs = ninputs 
        self.noutputs = noutputs
        self.test = test
        # Initialize parameters to default values
        self.ntrials = 1     # evaluation triala
        self.nttrials = 0    # post-evaluation trials
        self.maxsteps = 1000 # max number of steps (used from ERPolicy only)
        self.nhiddens = 50   # number of hiddens
        self.nhiddens2 = 0   # number of hiddens of the second layer 
        self.nlayers = 1     # number of hidden layers 
        self.bias = 0        # whether we have biases
        self.out_type = 2    # output type (1=logistic,2=tanh,3=linear,4=binary)
        self.architecture =0 # Feed-forward, recurrent, or full-recurrent network
        self.afunction = 2   # activation function
        self.nbins = 1       # number of bins 1=no-beans
        self.winit = 0       # weight initialization: Xavier, normc, uniform
        self.action_noise = 0# whether we apply noise to actions
        self.action_noise_range = 0.01 # action noise range
        self.normalize = 0   # Do not normalize observations
        self.clip = 0        # clip observation
        self.displayneurons=0# Gym policies can display or the robot or the neurons activations
        self.wrange = 1.0    # weight range, used in uniform initialization only
        # Read configuration file
        self.readConfig(filename)
        # Display info
        print("Evaluation: Episodes %d Test Episodes %d MaxSteps %d" % (self.ntrials, self.nttrials, self.maxsteps))
        # Initialize the neural network
        self.nn = net.PyEvonet(nrobots, heterogeneous, self.ninputs, (self.nhiddens * self.nlayers), self.noutputs, self.nlayers, self.nhiddens2, self.bias, self.architecture, self.afunction, self.out_type, self.winit, self.clip, self.normalize, self.action_noise, self.action_noise_range, self.wrange, self.nbins, low, high)
        # Initialize policy parameters
        self.nparams = self.nn.computeParameters()
        self.params = np.arange(self.nparams, dtype=np.float64)
        # Initialize normalization vector
        if (self.normalize == 1):
            self.normvector = np.arange(self.ninputs*2, dtype=np.float64)
        else:
            self.normvector = None
        # allocate neuron activation vector
        if (self.nbins == 1):
            self.nact = np.arange((self.ninputs + (self.nhiddens * self.nlayers) + self.noutputs) * nrobots, dtype=np.float64)
        else:
            self.nact = np.arange((self.ninputs + (self.nhiddens * self.nlayers) + (self.noutputs * self.nbins)) * nrobots, dtype=np.float64)            
        # Allocate space for observation and action
        self.ob = ob
        self.ac = ac
        # Copy pointers
        self.nn.copyGenotype(self.params)
        self.nn.copyInput(self.ob)
        self.nn.copyOutput(self.ac)
        self.nn.copyNeuronact(self.nact)
        if (self.normalize == 1):
            self.nn.copyNormalization(self.normvector)
        # Initialize weights
        self.nn.seed(self.seed)
        self.nn.initWeights()
        states = create_states()
        self.states_list = list(itertools.product(*states))

        test_states = create_test_states()
        self.test_list = list(itertools.product(*test_states))

         #= np.zeros(len(self.states_list))

        self.states_score = [np.array([0]) for _ in range(len(self.states_list))]
        self.vec_distributions = np.linspace(0,len(self.states_list),11,dtype=np.int32)
        self.choose = False



    def reset(self):
        self.nn.seed(self.seed)
        self.nn.initWeights()
        if (self.normalize == 1):
            self.nn.resetNormalizationVectors()

    def update_scores(self,trial_position,reward):
        if self.states_score[trial_position][0] == 0:
            self.states_score[trial_position] = np.array([reward])
        else:
            self.states_score[trial_position] = np.append(self.states_score[trial_position],reward)
            if len(self.states_score[trial_position]) > 5:
                self.states_score[trial_position] = np.delete(self.states_score[trial_position],0)

    def average_scores(self):
        self.ave = [np.mean(j) for j in self.states_score]
        self.sort_distributions = np.argsort(self.ave)
        #self.sort_positions = np.sort(ave)
        #self.get_positions()

    def bins_distributions(self):
        rescaled_ave = np.interp(self.ave,(np.min(self.ave),np.max(self.ave)),(0,1))

        bins = np.linspace(np.min(rescaled_ave), np.max(rescaled_ave), 11)
        bins = bins**2
        bins[-1] =1
        
        self.categorized_positions = []
        for j in range(1,11):
            self.categorized_positions.append(np.where((rescaled_ave>=bins[j-1])& (rescaled_ave<=bins[j]))[0])
            #print(len(np.where((self.ave>=bins[j-1])& (self.ave<bins[j]))[0]))

    def from_categories_get_positions(self):
        trials =[]
        for j in range(len(self.categorized_positions)):
            if len(self.categorized_positions[j]<10):
                tmp = self.categorized_positions[j][np.random.choice(len(self.categorized_positions[j]))]
            else:
                tmp = np.random.choice(len(self.states_list),1,replace=False)[0]
                
            trials.append(tmp)
        
        return trials

    def get_positions(self):
        trials = []
        for j in range(1,len(self.vec_distributions)):
            to_choose = self.sort_distributions[self.vec_distributions[j-1]:self.vec_distributions[j]]
            trials.append(to_choose[np.random.choice(self.vec_distributions[j]-self.vec_distributions[j-1],1)][0])
        return trials

    # virtual function, implemented in sub-classes
    def rollout(self, render=False, timestep_limit=None, seed=None):
                raise NotImplementedError

    def set_trainable_flat(self, x):
        self.params = np.copy(x)
        self.nn.copyGenotype(self.params)

    def get_trainable_flat(self):
        return self.params


    def readConfig(self, filename):
        # parse the [POLICY] section of the configuration file
        config = configparser.ConfigParser()
        config.read(filename)
        options = config.options("POLICY")
        for o in options:
          found = 0
          if (o == "ntrials"):
              self.ntrials = config.getint("POLICY","ntrials")
              found = 1
          if (o == "nttrials"):
              self.nttrials = config.getint("POLICY","nttrials")
              found = 1
          if (o == "maxsteps"):
              self.maxsteps = config.getint("POLICY","maxsteps")
              found = 1
          if (o == "nhiddens"):
              self.nhiddens = config.getint("POLICY","nhiddens")
              found = 1
          if (o == "nhiddens2"):
              self.nhiddens2 = config.getint("POLICY","nhiddens2")
              found = 1
          if (o == "nlayers"):
              self.nlayers = config.getint("POLICY","nlayers")
              found = 1
          if (o == "bias"):
              self.bias = config.getint("POLICY","bias")
              found = 1
          if (o == "out_type"):
              self.out_type = config.getint("POLICY","out_type")
              found = 1
          if (o == "nbins"):
              self.nbins = config.getint("POLICY","nbins")
              found = 1
          if (o == "afunction"):
              self.afunction = config.getint("POLICY","afunction")
              found = 1
          if (o == "architecture"):
              self.architecture = config.getint("POLICY","architecture")
              found = 1
          if (o == "winit"):
              self.winit = config.getint("POLICY","winit")
              found = 1
          if (o == "action_noise"):
              self.action_noise = config.getint("POLICY","action_noise")
              found = 1
          if (o == "action_noise_range"):
              self.action_noise_range = config.getfloat("POLICY","action_noise_range")
              found = 1
          if (o == "normalize"):
              self.normalize = config.getint("POLICY","normalize")
              found = 1
          if (o == "clip"):
              self.clip = config.getint("POLICY","clip")
              found = 1
          if (o == "wrange"):
              self.wrange = config.getint("POLICY","wrange")
              found = 1  
          if (found == 0):
              print("\033[1mOption %s in section [POLICY] of %s file is unknown\033[0m" % (o, filename))
              sys.exit()

    @property
    def get_seed(self):
        return self.seed

# Bullet use float values for observation and action vectors
# The policy communicate the pointer to the new vector each timestep since pyBullet create a new vector each time
# Use renderWorld to show neurons
class BulletPolicy(Policy):
    def __init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test):
        Policy.__init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test)                            
    
    # === Rollouts/training ===
    def rollout(self, ntrials, render=False, timestep_limit=None, seed=None):
        rews = 0.0
        steps = 0
        # initialize the render for showing the activation of the neurons
        if (self.test == 2):
            self.objs = np.arange(10, dtype=np.float64)   
            self.objs[0] = -1
            import renderWorld
        # To ensure replicability (we always pass a valid seed, even if fully-random evaluation is going to be run)
        if seed is not None:
            self.env.seed(seed)
            self.nn.seed(seed)
        # Loop over the number of trials
        for trial in range(ntrials):
            # if normalize=1, occasionally we store data for input normalization
            if self.normalize:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    normphase = 1
                    self.nn.normphase(1)
                else:
                    normphase = 0
            # Reset environment
            self.ob = self.env.reset()
            # Reset network
            self.nn.resetNet()
            # Reward for current trial
            crew = 0.0
            # Reset episode-reward and step counter for current trial
            rew = 0
            t = 0
            while t < self.maxsteps:
                # Copy the input in the network
                self.nn.copyInput(self.ob)
                # Activate network
                self.nn.updateNet()
                # Perform a step
                self.ob, r, done, _ = self.env.step(self.ac)
                rew += r
                t += 1
                if (self.test > 0):
                    if (self.test == 1):
                        self.env.render(mode="human")
                        time.sleep(0.05)
                    if (self.test == 2):
                        info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, r, rew)
                        renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)
                if done:
                    break
            if (self.test > 0):
                print("Trial %d Fit %.2f Steps %d " % (trial, rew, t))
            # if we normalize, we might need to stop store data for normalization
            if self.normalize and normphase > 0:
                self.nn.normphase(0)
            # Update steps
            steps += t
            rews += rew
        # Normalize reward by the number of trials
        rews /= ntrials
        if (self.test > 0 and ntrials > 1):
            print("Average Fit %.2f Steps %.2f " % (rews, steps/float(ntrials)))
        return rews, steps
    
# standard gym policy use double observation and action vectors and recreate the observation vector each step
# consequently we convert the observation vector in double and we communicate the pointer to evonet each step
# Use renderWorld to show neurons
class GymPolicy(Policy):
    def __init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test):
        Policy.__init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test)
    
    # === Rollouts/training ===
    def rollout(self, ntrials, render=False, timestep_limit=None, seed=None):
        rews = 0.0
        steps = 0
        # initialize the render for showing the activation of the neurons
        if (self.test == 2):
            import renderWorld
            self.objs = np.arange(10, dtype=np.float64)   
            self.objs[0] = -1 
        # To ensure replicability (we always pass a valid seed, even if fully-random evaluation is going to be run)
        if seed is not None:
            self.env.seed(seed)
            self.nn.seed(seed)
        # Loop over the number of trials
        for trial in range(ntrials):
            # if normalize=1, occasionally we store data for input normalization
            if self.normalize:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    normphase = 1
                    self.nn.normphase(1)
                else:
                    normphase = 0
            # Reset environment
            self.ob = self.env.reset()
            # Reset network
            self.nn.resetNet()
            # Reset episode-reward and step counter for current trial
            rew = 0.0
            t = 0
            while t < self.maxsteps:
                # Copy the input in the network
                self.nn.copyInput(np.float32(self.ob))
                # Activate network
                self.nn.updateNet()
                # Perform a step
                self.ob, r, done, _ = self.env.step(self.ac)
                # Append the reward
                rew += r
                t += 1
                if (self.test > 0):
                    if (self.test == 1):
                        self.env.render()
                        time.sleep(0.05)
                    if (self.test == 2):
                        info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, r, rew)
                        renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)
                if done:
                    break
            if (self.test > 0):
                print("Trial %d Fit %.2f Steps %d " % (trial, rew, t))
            # if we normalize, we might need to stop store data for normalization
            if self.normalize and normphase > 0:
                self.nn.normphase(0)
            # Update steps
            steps += t
            rews += rew
        # Normalize reward by the number of trials
        rews /= ntrials
        if (self.test > 0 and ntrials > 1):
            print("Average Fit %.2f Steps %.2f " % (rews, steps/float(ntrials)))
        return rews, steps

# standard gym policy use double observation and action vectors and recreate the observation vector each step
# consequently we convert the observation vector in double and we communicate the pointer to evonet each step
# in addition we convert the action vector into an integer since this policy is used for discrete output environment
# Use renderWorld to show neurons
class GymPolicyDiscr(Policy):
    def __init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test):
        Policy.__init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test)
    
    # === Rollouts/training ===
    def rollout(self, ntrials, render=False, timestep_limit=None, seed=None):
        rews = 0.0
        steps = 0
        # initialize the render for showing the activation of the neurons
        if (self.test == 2):
            import renderWorld
            self.objs = np.arange(10, dtype=np.float64)   
            self.objs[0] = -1 
        # To ensure replicability (we always pass a valid seed, even if fully-random evaluation is going to be run)
        if seed is not None:
            self.env.seed(seed)
            self.nn.seed(seed)
        # Loop over the number of trials
        for trial in range(ntrials):
            # if normalize=1, occasionally we store data for input normalization
            if self.normalize:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    normphase = 1
                    self.nn.normphase(1)
                else:
                    normphase = 0
            # Reset environment
            self.ob = self.env.reset()
            # Reset network
            self.nn.resetNet()
            # Reset episode-reward and step counter for current trial
            rew = 0.0
            t = 0
            while t < self.maxsteps:
                # Copy the input in the network
                self.nn.copyInput(np.float32(self.ob))
                # Activate network
                self.nn.updateNet()
                # Convert the action array into an integer
                action = np.argmax(self.ac)
                # Perform a step
                self.ob, r, done, _ = self.env.step(action)
                # Append the reward
                rew += r
                t += 1
                if render:
                    if (self.test == 1):
                        self.env.render()
                        time.sleep(0.05)
                    if (self.test == 2):
                        info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, r, rew)
                        renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)
                if done:
                    break
            if (self.test > 0):
                print("Trial %d Fit %.2f Steps %d " % (trial, rew, t))
            # if we normalize, we might need to stop store data for normalization
            if self.normalize and normphase > 0:
                self.nn.normphase(0)
            # Update steps
            steps += t
            rews += rew
        # Normalize reward by the number of trials
        rews /= ntrials
        if (self.test > 0 and ntrials > 1):
            print("Average Fit %.2f Steps %d " % (rews, steps/float(ntrials)))
        return rews, steps

# Evorobotpy policies use float observation and action vectors and do not re-allocate the observation vectors
# and use renderWorld to show robots and neurons
class ErPolicy(Policy):
    def __init__(self, env, ninputs, noutputs, low, high, ob, ac, done, filename, seed, nrobots, heterogeneous, test):
        Policy.__init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test)
        self.done = done
            
    # === Rollouts/training ===
    def rollout(self, ntrials, render=False, timestep_limit=None, seed=None):
        rews = 0.0
        steps = 0
        # To ensure replicability (we always pass a valid seed, even if fully-random evaluation is going to be run)
        if seed is not None:
            self.env.seed(seed)
            self.nn.seed(seed)
        # initialize the render for showing the behavior of the robot/s and the activation of the neurons
        if (self.test > 0):
            self.objs = np.arange(1000, dtype=np.float64) # the environment can contain up to 100 objects to be displayed  
            self.objs[0] = -1
            self.env.copyDobj(self.objs)
            import renderWorld
        if self.choose == False or ntrials>100:
            trials = np.random.choice(len(self.states_list),ntrials,replace=False)
        else:
            trials = self.from_categories_get_positions()#self.get_positions()

        # Loop over the number of trials
        for trial in range(ntrials):
            # if normalize=1, occasionally we store data for input normalization
            if self.normalize:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    normphase = 1
                    self.nn.normphase(1)
                else:
                    normphase = 0
            # Reset environment
            if ntrials > 100:
                self.env.reset()
            else:
                init_state = self.states_list[trials[trial]]
                self.env.startState(np.float32(init_state))

            # Reset network
            self.nn.resetNet()
            # Reset episode-reward and step counter for current trial
            rew = 0.0
            t = 0
            while t < self.maxsteps:
                # Activate network
                self.nn.updateNet()
                # Perform a step
                rew += self.env.step()
                t += 1
                # Render
                if (self.test > 0):
                    self.env.render()
                    info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, rew, rews)
                    renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)

                if self.done:
                    break
            if (self.test > 0):
                print("Trial %d Fit %.2f Steps %d " % (trial, rew, t))
            # if we normalize, we might need to stop store data for normalization
            if self.normalize and normphase > 0:
                self.nn.normphase(0)
            # Update steps
            steps += t
            rews += rew
            self.update_scores(trials[trial],reward=rew/1000)

        # Normalize reward by the number of trials
        rews /= ntrials
        if (self.test > 0 and ntrials > 1):
            print("Average Fit %.2f Steps %.2f " % (rews, steps/float(ntrials)))
        return rews, steps

    # === Rollouts/training ===
    def rollout_test(self):

        fit_list = []
        # Loop over the number of trials
        for trial in range(len(self.test_list)):
            # if normalize=1, occasionally we store data for input normalization
            if self.normalize:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    normphase = 1
                    self.nn.normphase(1)
                else:
                    normphase = 0
            # Reset environment
            # self.env.reset()

            init_state = self.test_list[trial]

            self.env.startState(np.float32(init_state))

            # Reset network
            self.nn.resetNet()
            # Reset episode-reward and step counter for current trial
            rew = 0.0
            t = 0
            while t < self.maxsteps:
                # Activate network
                self.nn.updateNet()
                # Perform a step
                rew += self.env.step()
                t += 1


                if self.done:
                    break
            fit_list.append(rew)
            # if we normalize, we might need to stop store data for normalization
            if self.normalize and normphase > 0:
                self.nn.normphase(0)


        return np.array(fit_list)
