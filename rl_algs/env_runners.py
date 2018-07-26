import types
from functools import reduce

import tensorflow as tf
import numpy as np
from typing import List
import scipy.signal

# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
    img = np.dot(rgb[:, :, :3], [0.299, 0.587, 0.114])
    return np.expand_dims(img, -1)

def pad_arrays(arrays: List) -> np.ndarray:
    shape = (len(arrays),) + reduce(max, (arr.shape for arr in arrays))
    padded = np.zeros(shape, dtype=arrays[0].dtype)
    for i, arr in enumerate(arrays):
        arr_slice = (i,) + tuple(slice(0, dim) for dim in arr.shape)  # type: ignore
        padded[arr_slice] = arr
    return padded

class Rollout:

        def __init__(self, partial_rollout):
            self.length = len(partial_rollout)
            self.keys = []
            self.obs = self._add_key_val('obs', partial_rollout.obs)
            self.act = self._add_key_val('act', partial_rollout.act)
            self.rew = self._add_key_val('rew', partial_rollout.rew)
            self.rew_in = self._add_key_val('rew_in', partial_rollout.rew_in)
            self.val = None
            self.episode_rew = np.sum(self.rew)

        def _add_key_val(self, key, val):
            if val is None:
                return None

            self.keys.append(key)
            val = np.squeeze(np.array(val))
            if self.length == 1:
                val = np.expand_dims(val, 0)
            return val

        def setVal(self, gamma):
            rewards = self.rew[::-1]
            self.val = scipy.signal.lfilter([1], [1, -gamma], rewards, axis=0)[::-1]
            self.keys.append('val')

        def __len__(self):
            return self.length

class BatchRollout(Rollout):
    
    def __init__(self, rollouts, variable_length=False):
        self.variable_length = variable_length
        if not variable_length:
            self.obs = np.array([roll.obs for roll in rollouts])
            self.act = np.array([roll.act for roll in rollouts])
        else:
            assert all(roll.obs.ndim == rollouts[0].obs.ndim for roll in rollouts), \
                'Cannot handle varying numbers of dimensions in observation'
            assert all(roll.act.ndim == rollouts[0].act.ndim for roll in rollouts), \
                'Cannot handle varying numbers of dimensions'
            self.obs = pad_arrays(roll.obs for roll in rollouts)
            self.act = pad_arrays(roll.act for roll in rollouts)

        self.rew = np.array([roll.rew for roll in rollouts])
        self.rew_in = np.array([roll.rew_in for roll in rollouts])
        self.val = None
        if hasattr(rollouts[0], 'val') and rollouts[0].val is not None:
            self.val = np.array([roll.val for roll in rollouts])

    def extend(self, rollout):
        if not isinstance(rollout, BatchRollout):
            raise TypeError('extend expected BatchRollout, received {}'.format(type(rollout)))
        if self.variable_length:
            raise NotImplementedError("Need to alter extend to work with variable lengths")
        self.obs = np.concatenate((self.obs, rollout.obs), 0)
        self.act = np.concatenate((self.act, rollout.act), 0)
        self.rew = np.concatenate((self.rew, rollout.rew), 0)
        self.rew_in = np.concatenate((self.rew_in, rollout.rew_in), 0)
        if self.val is not None:
            self.val = np.concatenate((self.val, rollout.val), 0)


class PartialRollout:

    def __init__(self):
        self.obs = None
        self.act = None
        self.rew = None
        self.rew_in = None
        self.length = 0
        self.episode_rew = 0

    def addObs(self, obs):
        if self.obs is None:
            self.obs = []
        self.obs.append(obs)
        self.length += 1

    def addAct(self, act):
        if self.act is None:
            self.act = []
        self.act.append(act)

    def addRew(self, rew):
        if self.rew is None:
            self.rew = []
        self.rew.append(rew)
        self.episode_rew += rew

    def addRewIn(self, rew):
        if self.rew_in is None:
            self.rew_in = []
        self.rew_in.append(rew)

    def finalize(self):
        assert not self.obs or len(self.obs) == self.length, 'Observation length does not match rollout length'
        assert not self.act or len(self.act) == self.length, 'Action length does not match rollout length'
        assert not self.rew or len(self.rew) == self.length, 'Reward length does not match rollout length'
        assert not self.rew_in or len(self.rew_in) == self.length, 'Reward input length does not match rollout length'
        return Rollout(self)

    def __len__(self):
        return self.length

class EnvironmentRunner:

    def __init__(self, env, agent, **kwargs):
        self._env = env
        self._agent = agent    
        self._episode_num = 0
        self._max_episode_steps = kwargs.get('max_episode_steps', float('inf'))
        self._pass_reward_to_agent = kwargs.get('pass_reward_to_agent', False)
        self._initialize_reward_from_environment = kwargs.get('initialize_reward_from_environment', False)

        modifyobs = kwargs.get('modifyobs', None)
        modifyreward = kwargs.get('modifyreward', None)

        if modifyobs in [None, False]:
            self._modifyobs = lambda obs: obs
        elif isinstance(modifyobs, types.FunctionType):
            self._modifyobs = modifyobs
        elif modifyobs in ['grayscale', 'greyscale']:
            self._modifyobs = rgb2gray
        else:
            raise ValueError("Unknown option for modifyobs argument. Received {}".format(modifyobs))

        if modifyreward in [None, False]:
            self._modifyreward = lambda rew: rew
        elif isinstance(modifyreward, types.FunctionType):
            self._modifyreward = modifyreward
        elif modifyreward == 'zeroone':
            self._modifyreward = lambda rew: 1 if rew > 0 else 0
        elif modifyreward == 'penalize':
            self._modifyreward = lambda rew: rew - 1
        else:
            raise ValueError("Unknown option for modifyreward argument. Received {}".format(modifyreward))
        
        self._done = False
        self.reset()

    def _prepareObs(self, obs):
        if tf.executing_eagerly():
            if obs.dtype == np.uint8:
                dtype = tf.uint8
            elif obs.dtype in [np.int32, np.int64]:
                dtype = tf.int32
            else:
                dtype = tf.float32

            obs = tf.constant(obs, dtype)
        obs = obs[None]
        if self._pass_reward_to_agent:
            rew = self._rew
            if tf.executing_eagerly():
                rew = tf.constant(rew, tf.float32)
            rew = rew[None]
            obs = (obs, rew)
        return obs

    def getRollout(self):
        while not self._done:
            self.step()

        return self._rollout.finalize()

    def step(self, obs=None, random=False):
        action = self.getAction(obs, random)
        return self.stepEnv(action)

    def getAction(self, obs=None, random=False):
        if obs is None:
            obs = self._obs

        if random:
            return np.random.randint(self._env.action_space.n)
        else:
            obs = self._prepareObs(obs)
            return self._agent.predict(obs)

    def stepEnv(self, action):
        if self._done:
            raise RuntimeError("Cannot step environment which is done. Call reset first.")

        self._rollout.addObs(self._obs)
        self._rollout.addRewIn(self._rew)

        obs, rew, done, _ = self._env.step(action)
        self._num_steps += 1
        if self._num_steps >= self._max_episode_steps:
            done = True

        self._obs = self._modifyobs(obs)
        self._rew = self._modifyreward(rew)
        self._done = done
        self._act = action

        self._rollout.addAct(self._act)
        self._rollout.addRew(self._rew)

        return self._done

    def reset(self):
        if self._done:
            self._episode_num += 1

        obs = self._env.reset()
        self._obs = self._modifyobs(obs)
        self._done = False
        self._act = None
        self._rew = 0. if not self._initialize_reward_from_environment else self._env._get_reward()
        self._num_steps = 0
        self._rollout = PartialRollout()

    def _get_printstr(self):
        printstr = []
        printstr.append('EPISODE: {:>7}'.format(self.episode_num))
        printstr.append('REWARD: {:>5.2f}'.format(self.episode_rew))
        printstr.append('NSTEPS: {:>5}'.format(self.episode_steps))
        return printstr

    @property
    def episode_rew(self):
        return self._rollout.episode_rew

    @property
    def episode_steps(self):
        return len(self._rollout)

    @property
    def episode_num(self):
        return self._episode_num

    @property
    def summary(self):
        return '\t' + ', '.join(self._get_printstr())

    @property
    def isDone(self):
        return self._done
    

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

class PGEnvironmentRunner(EnvironmentRunner):
    def __init__(self, env, agent, gamma, **kwargs):
        self._gamma = gamma
        super().__init__(env, agent, **kwargs)

    def getRollout(self):
        rollout = super().getRollout()
        rollout.setVal(self._gamma)
        return rollout

class VectorizedRunner:

    def __init__(self, runners, agent, pass_reward_to_agent=False):
        self._runners = runners
        self._current_runners = None
        self._agent = agent
        self._pass_reward_to_agent = pass_reward_to_agent
        self._done = False
        self._episode_num = 0
        self.reset()

    def _prepareObs(self, obs, rew=None):
        if tf.executing_eagerly():
            if obs.dtype == np.uint8:
                dtype = tf.uint8
            elif obs.dtype in [np.int32, np.int64]:
                dtype = tf.int32
            else:
                dtype = tf.float32
            obs = tf.constant(obs, dtype=dtype)

        if rew is not None:
            if tf.executing_eagerly():
                rew = tf.constant(rew, tf.float32)
            obs = (obs, rew)
        return obs

    def sampleRunners(self, n_runners):
        assert n_runners <= self.num_runners, 'Not enough environments to generate that many rollouts'
        indices = np.random.choice(self.num_runners, n_runners, replace=False)
        self._current_runners = [self._runners[i] for i in indices]

    def getRollouts(self):
        while not self._done:
            self.step()

        rollouts = [runner.getRollout() for runner in self._current_runners]
        return rollouts

    def step(self):
        actions = self.getAction()
        return self.stepEnv(actions)

    def getAction(self):
        obs = np.stack([runner._obs for runner in self._current_runners], 0)
        rew = None
        if self._pass_reward_to_agent:
            rew = np.array([runner._rew for runner in self._current_runners])
        obs = self._prepareObs(obs, rew)
        action = self._agent.predict(obs)
        action = [action[i] for i in range(len(self._current_runners))]
        return action

    def stepEnv(self, actions):
        if self._done:
            raise RuntimeError("Cannot step environment which is done. Call reset first.")
        done = False
        for runner, act in zip(self._current_runners, actions):
            runner.stepEnv(act)
            if runner.isDone:
                done = True

        self._done = done

        return done

    def reset(self):
        if self._current_runners is not None:
            for runner in self._current_runners:
                runner.reset()

            if self._done:
                self._episode_num += len(self._current_runners)

        self._current_runners = None
        self._done = False

    @property
    def num_runners(self):
        return len(self._runners)

    @property
    def episode_rew(self):
        if self._current_runners is None:
            return None

        return np.mean(runner.episode_rew for runner in self._current_runners)

    @property
    def episode_steps(self):
        if self._current_runners is None:
            return None

        return np.mean(runner.episode_steps for runner in self._current_runners)

    @property
    def episode_num(self):
        return self._episode_num

    @property
    def summary(self):
        return '\t' + ', '.join(self._get_printstr())

    @property
    def isDone(self):
        return self._done
    

