import os
from operator import attrgetter
from typing import Optional, Dict, Any, Type, Union  # noqa: F401
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as pkl

from tensorflow.keras import Model
import tensorflow.keras.backend as K
import gym
from baselines.common.distributions import make_pdtype
from baselines.common.tf_util import adjust_shape, ALREADY_INITIALIZED
from baselines.common.input import observation_placeholder
import h5py

from .NatureCNN import NatureCNN
from .Policy import Policy  # noqa: F401
from .StandardPolicy import StandardPolicy
from .LSTMPolicy import LSTMPolicy
from .RMCPolicy import RMCPolicy

receptive_field = np.load('/home/roshan_rao/projects/spatial-transformer/visualize/receptive_field.npy')
contrib = receptive_field / np.sum(receptive_field, (0, 1), keepdims=True)


def show_attention(batch: int, images, attention) -> None:
    plt.close('all')
    num_heads = attention.shape[1]
    mem_slots = attention.shape[2]
    #  attn shape: [num_heads, mem_slots, 84, 84]
    attn = np.sum(attention[batch, :, :, None, None] * contrib, (-1, -2))
    # fig = plt.figure(figsize=(mem_slots, num_heads))
    plt.figure()
    full_image = np.tile(images[batch, :, :, -1], (mem_slots, num_heads, 1))
    full_attn = attn.swapaxes(1, 2).reshape((num_heads * 84, mem_slots * 84)).swapaxes(0, 1)
    plt.imshow(full_image)
    plt.contourf(full_attn)

    plt.show()


class BaselinesPolicyWrapper(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    policy_num: int = 0

    def __init__(self,
                 env: gym.Env,
                 policy: Model,
                 observations: tf.Tensor,
                 estimate_q: bool = False,
                 sess: Optional[tf.Session] = None,
                 extra_inputs: Optional[Dict[str, Any]] = None) -> None:
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """
        BaselinesPolicyWrapper.policy_num += 1
        self.this_policy = BaselinesPolicyWrapper.policy_num
        if extra_inputs is None:
            extra_inputs = {}
        output = policy(observations, **extra_inputs)

        self.policy = policy
        self.X = observations
        self.initial_state = None
        self.__dict__.update(output)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)
        self.test = []

        # self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)
        self.pd = policy.pd

        # Calculate the neg log of our probability
        self.sess = sess or tf.get_default_session()

        # self.load_from_lrl()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.vf = self.q  # type: ignore

    def load_from_lrl(self):
        weights = h5py.File('weights.h5')
        embedding_prefix = 'vqa_image_embedding/lrl_model/vqa_image_embedding/'
        rmc_prefix = 'rmc_memory/lrl_model/rmc_memory/relational_memory_core'
        # import IPython
        # IPython.embed()
        def get_weights(prefix):
            all_weights = []

            def _recursive_get(prefix):
                if isinstance(weights[prefix], h5py.Dataset):
                    all_weights.append(np.asarray(weights[prefix]))
                else:
                    for name in weights[prefix]:
                        newprefix = '/'.join([prefix, name])
                        _recursive_get(newprefix)

            _recursive_get(prefix)
            return all_weights

        def get_variables(layer):
            all_variables = []

            def _recursive_get(layer):
                if hasattr(layer, 'layers'):
                    for sublayer in sorted(filter(lambda l: l.variables, layer.layers), key=attrgetter('name')):
                        _recursive_get(sublayer)
                else:
                    all_variables.extend(list(sorted(layer.variables, key=attrgetter('name'))))

            _recursive_get(layer)
            return all_variables

        def set_weights(layer, prefix):
            variables = get_variables(layer)
            weights = get_weights(prefix)
            assert len(variables) == len(weights), \
                'Trying to load weight prefix with {} layers into a model with {} layers'.format(
                    len(weights), len(variables))
            for v, w in zip(variables, weights):
                if v.shape != w.shape:
                    print(v.name, v.shape, w.shape)
            K.batch_set_value(zip(variables, weights))
            ALREADY_INITIALIZED.update(variables)

        set_weights(self.policy.embedding_model, embedding_prefix)
        set_weights(self.policy.cell, rmc_prefix)

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask
            (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state,
            negative log likelihood of the action under current policy parameters) tuple
        """
        a, v, state, neglogp, attn = self._evaluate(
            [self.action, self.vf, self.state, self.neglogp, self.policy.attention], observation, **extra_feed)

        if self.this_policy == 1 and len(self.test) < 50:
            self.test.append((observation, attn))
            if len(self.test) >= 50:
                with open('test.pkl', 'wb') as f:
                    pkl.dump(self.test, f)

        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask
                        (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        self.policy.save_weights(os.path.join(save_path, 'weights.h5'))

    def load(self, load_path):
        self.policy.load_weights(os.path.join(load_path, 'weights.h5'))


class BaselinesPolicyFnWrapper:

    def __init__(self,
                 env: gym.Env,
                 policy_network,
                 normalize_observations: bool = False,
                 estimate_q: bool = False) -> None:

        self.env = env
        self.policy_network = policy_network
        self.normalize_observations = normalize_observations
        self.estimate_q = estimate_q

        policy_type = StandardPolicy  # type: Type[Union[StandardPolicy, LSTMPolicy, RMCPolicy]]
        extra_args = {}  # type: Dict[str, Any]
        self.recurrent = False
        use_rmc = False

        if 'lstm' in policy_network:
            policy_type = LSTMPolicy
            extra_args = {'lstm_cell_size': 128}
            self.recurrent = True
        elif 'rmc' in policy_network:
            policy_type = RMCPolicy
            extra_args = {
                'mem_slots': 3,
                'mem_size': 64,
                'n_heads': 4,
                'treat_input_as_sequence': True,
                'use_cross_attention': False}
            self.recurrent = True
            use_rmc = True

        embedding_model = NatureCNN(use_rmc)

        policy = policy_type(
            env.observation_space,
            env.action_space,
            embedding_model,
            model_dim=512,
            n_layers_logits=0,
            n_layers_value=0,
            take_greedy_actions=estimate_q,
            normalize_observations=normalize_observations,
            **extra_args)

        self.policy = policy

    def __call__(self, nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = self.env.observation_space
        observations = observ_placeholder if observ_placeholder is not None else \
            observation_placeholder(ob_space, batch_size=nbatch)

        if self.recurrent:
            mask = tf.placeholder(tf.float32, [nbatch])
            nenv = nbatch // nsteps
            assert nenv > 0, 'nbatch cannot be less than nsteps'
            initial_state = tf.placeholder(tf.float32, [nenv, self.policy.state_size])
            extra_inputs = {
                'initial_state': initial_state,
                'mask': mask,
                'nenv': nenv,
                'nsteps': nsteps}
        else:
            extra_inputs = {}

        policy = BaselinesPolicyWrapper(
            self.env, self.policy, observations, self.estimate_q, sess, extra_inputs=extra_inputs)

        return policy
