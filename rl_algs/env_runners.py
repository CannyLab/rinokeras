import sys
import types
from collections import defaultdict

import numpy as np
import tensorflow as tf
import gym
import gym_gridworld
import scipy.signal

from .policies.Policy import Policy

# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
    img = np.dot(rgb[:,:,:3], [0.299, 0.587, 0.114])
    return np.expand_dims(img, -1)

class EnvironmentRunner(object):
    def __init__(self, env, agent, **kwargs):
        assert isinstance(agent, Policy), "Agent must be a subclass of Policy. Received {}".format(type(agent))

        self._env = env
        self._agent = agent    
        self._episode_num = 0
        self._return_activations = kwargs.get('return_activations', False)
        self._max_episode_steps = kwargs.get('max_episode_steps', float('inf'))

        modifyobs = kwargs.get('modifyobs', None)
        modifyreward = kwargs.get('modifyreward', None)

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
        
        self._done = False
        self.reset()

    def get_rollout(self):
        if self._done:
            self.reset()

        while not self._done:
            self.step()

        return self._rollout

    def step(self, obs=None, random=False):
        if self._done:
            raise RuntimeError("Cannot step environment which is done. Call reset first.")
        self._rollout['obs'].append(self._obs)
        if obs is None:
            obs = self._obs

        # Get action
        if random:
            action = np.random.randint(self._env.action_space.n)
        else:
            pred = self._agent.predict(obs, return_activations=self._return_activations)
            if self._return_activations:
                action, qval, activs = pred
            else:
                action = pred
            self._num_agent_actions += 1
        self._num_steps += 1

        # Step the environment
        obs, rew, done, _ = self._env.step(action)
        if self._num_steps >= self._max_episode_steps:
            done = True

        self._obs = self._modifyobs(obs)
        self._rew = self._modifyreward(rew)
        self._done = done
        self._act = action
        
        self._rollout['act'].append(self._act)
        self._rollout['rew'].append(self._rew)
        if self._return_activations:
            self._qval = None if random else qval
            self._activs = None if random else activs
            self._rollout['qval'].append(self._qval)
            self._rollout['activs'].append(self._activs)

        self._episode_rew += self._rew

        return self._done

    def reset(self):
        if self._done:
            self._episode_num += 1

        obs = self._env.reset()
        self._obs = self._modifyobs(obs)
        self._done = False
        self._act = None
        self._rew = None
        self._qval = None
        self._activs = None
        self._num_steps = 0
        self._num_agent_actions = 0
        self._episode_rew = 0
        self._rollout = defaultdict(lambda : [])

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

    @property
    def summary(self):
        printstr = []
        printstr.append('EPISODE: {:>7}'.format(self._episode_num))
        printstr.append('REWARD: {:>5}'.format(self._episode_rew))
        printstr.append('NSTEPS: {:>5}'.format(self._num_steps))
        printstr.append('PERCENT_AGENT: {:>6.2f}'.format(100 * self._num_agent_actions / self._num_steps))
        return '\t' + ', '.join(printstr)

class DQNEnvironmentRunner(EnvironmentRunner):
    def __init__(self, env, agent, replay_buffer, **kwargs):
        self._replay_buffer = replay_buffer
        super().__init__(env, agent, **kwargs)

    def step(self, epsilon=0):
        idx = self._replay_buffer.store_frame(self._obs)
        obs = self._replay_buffer.encode_recent_observation()[None]
        takerandom = np.random.random() < epsilon
        super().step(obs, takerandom)
        self._replay_buffer.store_effect(idx, self._act, self._rew, self._done)
        return self._done

    @property
    def summary(self):
        printstr = []
        printstr.append('EPISODE: {:>7}'.format(self._episode_num))
        printstr.append('REWARD: {:>5}'.format(self._episode_rew))
        printstr.append('NSTEPS: {:>5}'.format(self._num_steps))
        printstr.append('PERCENT_AGENT: {:>6.2f}'.format(100 * self._num_agent_actions / self._num_steps))
        return '\t' + ', '.join(printstr)

class PGEnvironmentRunner(EnvironmentRunner):
    def __init__(self, env, agent, gamma, **kwargs):
        self._gamma = gamma
        super().__init__(env, agent, **kwargs)

    def get_rollout(self):
        while not self._done:
            self.step(self._obs[None])

        for key in self._rollout:
            self._rollout[key] = np.squeeze(np.array(self._rollout[key]))
            if self._num_steps == 1:
                self._rollout[key] = np.expand_dims(self._rollout[key], 0)
        # compute values
        self._rollout['val'] = scipy.signal.lfilter([1], [1, -self._gamma], self._rollout['rew'][::-1], axis=0)[::-1]

        if self._return_activations:
            # normalize baseline to vals
            self._rollout['baseline'] -= self._rollout['baseline'].mean()
            self._rollout['baseline'] /= self._rollout['baseline'].std() + 1e-10 # deal with zero std
            self._rollout['baseline'] *= self._rollout['val'].std() + 1e-10 # deal with zero std
            self._rollout['baseline'] += self._rollout['val'].mean()

            # compute and normalize advantages
            self._rollout['adv'] = self._rollout['val'] - self._rollout['baseline']
            self._rollout['adv'] -= self._rollout['adv'].mean()
            self._rollout['adv'] /= self._rollout['adv'].std() + 1e-10

            # normalize vals to 0/1
            self._rollout['val'] -= self._rollout['val'].mean()
            self._rollout['val'] /= self._rollout['val'].std() + 1e-10

        return dict(self._rollout) # cast to dict so keys will error

    @property
    def summary(self):
        printstr = []
        printstr.append('EPISODE: {:>7}'.format(self._episode_num))
        printstr.append('REWARD: {:>5}'.format(self._episode_rew))
        printstr.append('NSTEPS: {:>5}'.format(self._num_steps))
        return '\t' + ', '.join(printstr)

    def reset(self):
        self._val = None
        super().reset()

