import sys
import logging
import types
from collections import defaultdict

import numpy as np
import tensorflow as tf
import gym
import gym_gridworld

from rl_algs.Policy import Policy

# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
    img = np.dot(rgb[:,:,:3], [0.299, 0.587, 0.114])
    return np.expand_dims(img, -1)

class EnvironmentRunner(object):
    def __init__(self, env, agent, modifyobs=None, modifyreward=None, verbose=False):
        assert isinstance(agent, Policy), "Agent must be a subclass of Policy. Received {}".format(type(agent))

        self._env = env
        self._agent = agent    
        self._episode_num = 0
        self._verbose = verbose

        if modifyobs in [None, False]:
            self._modifyobs = lambda obs : obs
        elif isinstance(modifyobs, types.FunctionType):
            self._modifyobs = modifyobs
        elif modifyobs in ['grayscale', 'greyscale']:
            self._modifyobs = rgb2gray
        else:
            raise ValueError("Unknown option for modifyobs argument. Received {}".format(modifyobs))

        if modifyreward in [None, False]:
            self._modifyreward = lambda rew : rew
        elif isinstance(modifyreward, types.FunctionType):
            self._modifyreward = modifyreward
        elif modifyreward == 'zeroone':
            self._modifyreward = lambda rew : 1 if rew > 0 else 0
        elif modifyreward == 'penalize':
            self._modifyreward = lambda rew : rew - 1
        else:
            raise ValueError("Unknown option for modifyrew argument. Received {}".format(modifyrew))

        self._obs = None
        self._rew = None
        self._act = None
        self._done = False
        self._episode_rew = 0
        self._num_steps = 0
        self._num_agent_actions = 0

        self.reset()

    def get_rollout(self):
        rollout = defaultdict(lambda : [])

        while not self._done:
            rollout['obs'].append(self._obs)
            super().step(obs, False)
            rollout['act'].append(self._act)
            rollout['rew'].append(self._rew)

        return rollout

    def step(self, obs=None, random=False):
        if self._done:
            raise RuntimeError("Cannot step environment which is done. Call reset first.")
        if obs is None:
            obs = self._obs

        # Get action
        if random:
            action = np.random.randint(self._env.action_space.n)
        else:
            action = self._agent.predict(obs)
            self._num_agent_actions += 1

        # Step the environment
        obs, rew, done, _ = self._env.step(action)
        self._obs = self._modifyobs(obs)
        self._rew = self._modifyreward(rew)
        self._done = done
        self._act = action
        self._num_steps += 1

        self._episode_rew += self._rew

        return self._done

    def reset(self):
        if self._done:
            if self._verbose:
                printstr = ''
                printstr += '\tReward: {:>5}'.format(self._episode_rew)
                printstr += ', NSTEPS: {:>5}'.format(self._num_steps)
                printstr += ', PERCENT_AGENT: {:>6.2f}'.format(100 * self._num_agent_actions / self._num_steps)
                logging.info(printstr)
            self._episode_num += 1

        obs = self._env.reset()
        self._obs = self._modifyobs(obs)
        self._done = False
        self._act = None
        self._rew = None
        self._num_steps = 0
        self._num_agent_actions = 0
        self._episode_rew = 0

    @property
    def episode_rew(self):
        return self._episode_rew

    @property
    def episode_steps(self):
        return self._num_steps

    @property
    def num_agent_action(self):
        return self._num_agent_actions

    @property
    def episode_num(self):
        return self._episode_num

class DQNEnvironmentRunner(EnvironmentRunner):
    def __init__(self, env, agent, replay_buffer, modifyobs=False, modifyreward=None, verbose=False):
        self._replay_buffer = replay_buffer
        super().__init__(env, agent, modifyobs, modifyreward, verbose)

    def step(self, epsilon):
        idx = self._replay_buffer.store_frame(self._obs)
        obs = self._replay_buffer.encode_recent_observation()[None]
        takerandom = np.random.random() < epsilon
        super().step(obs, takerandom)
        self._replay_buffer.store_effect(idx, self._act, self._rew, self._done)
        return self._done

# PGEnvironmentRunner -> should return values, advantages

# ActivationEnvironmentRunner -> should return agent activations



