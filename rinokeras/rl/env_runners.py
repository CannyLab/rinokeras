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
    arrays = list(arrays)
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

        def set_state_values(self, gamma):
            rewards = self.rew[::-1]
            self.val = scipy.signal.lfilter([1], [1, -gamma], rewards, axis=0)[::-1]
            self.keys.append('val')

        def __len__(self):
            return self.length


class BatchRollout(Rollout):

    def __init__(self, rollouts, variable_length=False, keep_as_separate_rollouts=False):

        def join_op(arrays):
            if keep_as_separate_rollouts and not variable_length:
                return np.stack(arrays, axis=0)
            elif keep_as_separate_rollouts and variable_length:
                return pad_arrays(arrays)
            else:
                return np.concatenate(arrays, axis=0)

        self.variable_length = variable_length
        if variable_length and keep_as_separate_rollouts:
            assert all(roll.obs.ndim == rollouts[0].obs.ndim for roll in rollouts), \
                'Cannot handle varying numbers of dimensions in observation'
            assert all(roll.act.ndim == rollouts[0].act.ndim for roll in rollouts), \
                'Cannot handle varying numbers of dimensions'

        self.obs = join_op([roll.obs for roll in rollouts])
        self.act = join_op([roll.act for roll in rollouts])
        self.rew = join_op([roll.rew for roll in rollouts])
        self.rew_in = join_op([roll.rew_in for roll in rollouts])
        self.val = None
        if hasattr(rollouts[0], 'val') and rollouts[0].val is not None:
            self.val = join_op([roll.val for roll in rollouts])

        self.seqlens = np.array([roll.length for roll in rollouts], dtype=np.int32)
        self.episode_rew = np.array([roll.episode_rew for roll in rollouts])

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

    def add_obs(self, obs):
        if self.obs is None:
            self.obs = []
        self.obs.append(obs)
        self.length += 1

    def add_act(self, act):
        if self.act is None:
            self.act = []
        self.act.append(act)

    def add_rew(self, rew):
        if self.rew is None:
            self.rew = []
        self.rew.append(rew)
        self.episode_rew += rew

    def add_rew_in(self, rew):
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
        self._episode_num = 1
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

        # self._reward_summary = tf.summary.scalar('Reward')
        # self._steps_summary = tf.summary.scalar('Num Steps')
        # self._summary_ops = tf.summary.merge([self._reward_summary, self._steps_summary])

        self._done = False
        self.reset()

    def _prepare_obs(self, obs):
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

    def __next__(self):
        self.reset()
        return self.get_rollout()

    def get_rollout(self):
        while not self._done:
            self.step()

        return self._rollout.finalize()

    def step(self, obs=None, random=False):
        action = self.get_action(obs, random)
        return self.step_env(action)

    def get_action(self, obs=None, random=False):
        if obs is None:
            obs = self._obs

        if random:
            return np.random.randint(self._env.action_space.n)
        else:
            obs = self._prepare_obs(obs)
            return self._agent.predict(obs)

    def step_env(self, action):
        if self._done:
            raise RuntimeError("Cannot step environment which is done. Call reset first.")

        self._rollout.add_obs(self._obs)
        self._rollout.add_rew_in(self._rew)

        obs, rew, done, _ = self._env.step(action)
        self._num_steps += 1
        if self._num_steps >= self._max_episode_steps:
            done = True

        self._obs = self._modifyobs(obs)
        self._rew = self._modifyreward(rew)
        self._done = done
        self._act = action

        self._rollout.add_act(self._act)
        self._rollout.add_rew(self._rew)

        return self._done

    def reset(self):
        if self._done:
            self._episode_num += 1

        obs = self._env.reset()
        self._obs = self._modifyobs(obs)
        self._done = False
        self._act = None
        self._rew = 0. if not self._initialize_reward_from_environment else self._env.get_reward()
        self._num_steps = 0
        self._agent.clear_memory()
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
        # sess = tf.get_default_session()
        # episode_summary = sess.run()
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

    def get_rollout(self):
        rollout = super().get_rollout()
        rollout.set_state_values(self._gamma)
        return rollout


class VectorizedRunner:

    def __init__(self, runners, agent, pass_reward_to_agent=False, variable_length=False):
        self._runners = runners
        self._current_runners = None
        self._agent = agent
        self._pass_reward_to_agent = pass_reward_to_agent
        self._done = False
        self._episode_num = 0
        self._variable_length = variable_length
        self.reset()

    def _prepare_obs(self, obs, rew=None):
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

    def get_rollouts(self):
        while not self._done:
            self.step()

        rollouts = [runner.get_rollout() for runner in self._current_runners]
        return rollouts

    def step(self):
        actions = self.get_action()
        return self.step_env(actions)

    def get_action(self):
        if self._variable_length:
            obs = pad_arrays(runner._obs for runner in self._current_runners)
        else:
            obs = np.stack([runner._obs for runner in self._current_runners], 0)
        rew = None
        if self._pass_reward_to_agent:
            rew = np.array([runner._rew for runner in self._current_runners])
        obs = self._prepare_obs(obs, rew)
        if self._variable_length:
            seqlens = np.array([runner._obs.shape[0] for runner in self._current_runners], dtype=np.int32)
            if tf.executing_eagerly():
                seqlens = tf.constant(seqlens, dtype=tf.int32)
            action = self._agent.predict(obs, padding_mask=seqlens)
        else:
            action = self._agent.predict(obs)
        action = [action[i] for i in range(len(self._current_runners))]
        if self._variable_length:
            action = [act[0, :seqlen] for act, seqlen in zip(action, seqlens)]
        return action

    def step_env(self, actions):
        if self._done:
            raise RuntimeError("Cannot step environment which is done. Call reset first.")
        done = False
        for runner, act in zip(self._current_runners, actions):
            runner.step_env(act)
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
        return '\t' + ', '.join(self._get_printstr())  # TODO: VectorizedRunner has no '_get_printstr' member

    @property
    def isDone(self):
        return self._done
