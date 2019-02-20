import os
from typing import Optional, Dict, Any, Type, Union  # noqa: F401
import tensorflow as tf

from tensorflow.keras import Model
import gym
from baselines.common.distributions import make_pdtype
from baselines.common.tf_util import adjust_shape
from baselines.common.input import observation_placeholder

from .NatureCNN import NatureCNN
from .Policy import Policy  # noqa: F401
from .StandardPolicy import StandardPolicy
from .LSTMPolicy import LSTMPolicy
from .RMCPolicy import RMCPolicy


class BaselinesPolicyWrapper(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

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

        if extra_inputs is None:
            extra_inputs = {}
        output = policy(observations, **extra_inputs)
        self.policy = policy
        self.X = observations
        self.initial_state = None
        self.__dict__.update(output)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)

        # self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)
        self.pd = policy.pd

        # Calculate the neg log of our probability
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.vf = self.q  # type: ignore

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
        a, v, state, neglogp = self._evaluate(
            [self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
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
                'mem_slots': 10,
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
