import argparse
import itertools

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import numpy as np
import gym
# import seaborn as sns

from rinokeras.rl.policies import StandardPolicy, LSTMPolicy
from rinokeras.rl.trainers import PolicyGradient, PPO
from rinokeras.rl.env_runners import PGEnvironmentRunner, BatchRollout
from rinokeras.train import TrainGraph

parser = argparse.ArgumentParser('Rinokeras RL Example Script')
parser.add_argument('--env', type=str, default='CartPole-v0', help='Which gym environment to run on')
parser.add_argument('--policy', type=str, choices=['standard', 'lstm'], default='standard',
                    help='Which type of policy to run')
parser.add_argument('--alg', type=str, choices=['vpg', 'ppo'], default='vpg',
                    help='Which algorithm to use to train the agent')
parser.add_argument('--logstd', type=float, default=0, help='initial_logstd')
args = parser.parse_args()

env = gym.make(args.env)
policies = {
    'standard': StandardPolicy,
    'lstm': LSTMPolicy}
algorithms = {
    'vpg': PolicyGradient,
    'ppo': PPO}

# Parameters/Hyperparameters
discrete = not isinstance(env.action_space, gym.spaces.Box)
action_shape = (env.action_space.n,) if discrete else env.action_space.shape
model_dim = 64
gamma = 0.95
n_rollouts_per_batch = 10
n_updates_per_batch = 1 if args.alg == 'vpg' else 3
embedding_model = Dense(model_dim)

# Placeholders
obs_ph = Input((None,) + env.observation_space.shape)
act_ph = Input((None,) + (() if discrete else env.action_space.shape), dtype=tf.int32 if discrete else tf.float32)
val_ph = Input((None,))
seqlen_ph = Input((), dtype=tf.int32)

# Setup policy, experiment, graph
policy = policies[args.policy](
    action_shape, 'discrete' if discrete else 'continuous', embedding_model, model_dim,
    initial_logstd=args.logstd, n_layers_logits=1, n_layers_value=1, take_greedy_actions=False)

experiment = algorithms[args.alg](policy, distribution_strategy=tf.contrib.distribute.OneDeviceStrategy('/cpu:0'))
graph = TrainGraph.from_experiment(experiment, (obs_ph, act_ph, val_ph, seqlen_ph))

runner = PGEnvironmentRunner(env, policy, gamma)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

all_rewards = []

# Do Training
for t in itertools.count():
    rollouts = []
    for _ in range(n_rollouts_per_batch):
        rollouts.append(next(runner))  # type: ignore

    batch_rollout = BatchRollout(rollouts, variable_length=True, keep_as_separate_rollouts=True)

    if args.alg == 'ppo':
        experiment.update_old_model()
    for _ in range(n_updates_per_batch):
        loss = graph.run('update', (batch_rollout.obs, batch_rollout.act, batch_rollout.val, batch_rollout.seqlens))

    mean_episode_reward = np.mean(batch_rollout.episode_rew)
    all_rewards.append(mean_episode_reward)
    mean_episode_steps = np.mean(batch_rollout.seqlens)
    current_episode_num = runner.episode_num

    printstr = []
    printstr.append('EPISODE: {:>7}'.format(current_episode_num))
    printstr.append('MEAN REWARD: {:>6.1f}'.format(mean_episode_reward))
    printstr.append('MEAN EPISODE STEPS: {:>5}'.format(mean_episode_steps))
    print(', '.join(printstr))

    if t > 500:
        break

np.save('-'.join([args.env, args.policy, args.alg, 'logstd=' + str(args.logstd)]) + '.npy', np.array(all_rewards))
