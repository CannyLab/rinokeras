import sys
import types
from collections import defaultdict

import numpy as np
import tensorflow as tf
import gym
import gym_gridworld

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
        rollout = defaultdict(lambda : [])

        while not self._done:
            rollout['obs'].append(self._obs)
            self.step()
            rollout['act'].append(self._act)
            rollout['rew'].append(self._rew)
            if self._return_activations:
                rollout['qval'].append(self._qval)
                rollout['activs'].append(self._activs)

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
            pred = self._agent.predict(obs, return_activations=self._return_activations)
            if self._return_activations:
                action, qval, activs = pred
            else:
                action = pred
            self._num_agent_actions += 1

        # Step the environment
        obs, rew, done, _ = self._env.step(action)
        self._obs = self._modifyobs(obs)
        self._rew = self._modifyreward(rew)
        self._done = done
        self._act = action
        if self._return_activations:
            self._qval = qval
            self._activs = activs
        self._num_steps += 1

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
        return ''

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
        rollout = defaultdict(lambda : [])

        while not self._done:
            rollout['obs'].append(self._obs)
            super().step(obs, False)
            rollout['act'].append(self._act)
            rollout['rew'].append(self._rew)
            rollout['baseline'].append(self._val)

        rollout['val'].append(self._val)
        for t in reversed(range(self._num_steps - 1)):
            rollout['val'].append(rollout['rew'][t] + self._gamma * rollout['val'][-1])

        rollout['val'].reverse()

        for key in rollout:
            rollout[key] = np.array(rollout[key])

        rollout['adv'] = rollout['val'] - rollout['baseline']
        rollout['adv'] = (rollout['adv'] - rollout['adv'].mean()) / rollout['adv'].std()

        return rollout

    def step(self):
        if self._done:
            raise RuntimeError("Cannot step environment which is done. Call reset first.")
        obs = self._obs

        action, val = self._agent.predict(obs)

        # Step the environment
        obs, rew, done, _ = self._env.step(action)
        self._obs = self._modifyobs(obs)
        self._rew = self._modifyreward(rew)
        self._val = val
        self._done = done
        self._act = action
        self._num_steps += 1

        self._episode_rew += self._rew

        return self._done

    @property
    def summary(self):
        printstr = ''
        printstr += '\tReward: {:>5}'.format(self._episode_rew)
        printstr += ', NSTEPS: {:>5}'.format(self._num_steps)
        return printstr

    def reset(self):
        self._val = None
        super().reset()

