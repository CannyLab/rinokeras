import argparse
parser = argparse.ArgumentParser(description='Reinforcement Learning Library for Research Purposes')
parser.add_argument('env', action='store', type=str, help='environment to run on')
parser.add_argument('alg', action='store', choices=['dqn', 'ppo', 'pg'], help='algorithm to run')
parser.add_argument('--device', action='store', type=int, default=0, choices=[0, 1], help='which gpu device to run on')
parser.add_argument('--policy', action='store', default='standard', choices=['standard', 'lstm', 'random'], help='which type of policy to run')
parser.add_argument('--gamma', action='store', type=float, default=0.99, help='discount factor')
parser.add_argument('--benchmark', action='store', type=str, default=None, help='where to save results for benchmarking')
parser.add_argument('--loglevel', action='store', choices=['debug', 'info', 'warning'], default='info', help='hide debugging information')
parser.add_argument('--max-episode-steps', action='store', type=int, default=-1, help='maximum number of steps to take')
parser.add_argument('--max-learning-steps', action='store', type=int, default=5000, help='maximum number of learning steps')
parser.add_argument('--eager', action='store_true', help='run in eager mode')
args = parser.parse_args()

import gym
env = gym.make(args.env)

import os
import logging
import itertools
import pickle as pkl

import numpy as np
import tensorflow as tf
if args.eager:
    import tensorflow.contrib.eager as tfe
    tf.enable_eager_execution()

if not args.eager:
    from rl_algs.policies import StandardPolicy, LSTMPolicy, RandomPolicy
    from rl_algs.trainers import DQNTrainer, PGTrainer, PPOTrainer
else:
    from rl_algs.eager.policies import StandardPolicy, LSTMPolicy
    from rl_algs.eager.trainers import PGTrainer, PPOTrainer
from rl_algs.utils import ReplayBuffer, PiecewiseSchedule
from rl_algs.env_runners import PGEnvironmentRunner, DQNEnvironmentRunner

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

discrete = not isinstance(env.action_space, gym.spaces.Box)
ob_shape = env.observation_space.shape
ac_shape = env.action_space.n if discrete else env.action_space.shape

action_method = 'greedy' if args.alg == 'dqn' else 'sample'

benchmark_results = {'args' : args}

policy = None
if args.policy == 'standard':
    policy = StandardPolicy(ob_shape,
                            ac_shape,
                            discrete,
                            action_method=action_method,
                            embedding_architecture=None,
                            value_architecture=[(64,)])
elif args.policy == 'lstm':
    policy = LSTMPolicy(ob_shape,
                        ac_shape,
                        discrete,
                        action_method=action_method,
                        embedding_architecture=None,
                        value_architecture=[(64,)])
elif args.policy == 'random':
    policy = RandomPolicy(ob_shape, ac_shape, discrete)

benchmark_results['embedding_architecture'] = policy._embedding_architecture
benchmark_results['logit_architecture'] = policy._logit_architecture
benchmark_results['value_architecture'] = policy._value_architecture
benchmark_results['lstm_cell_size'] = getattr(policy, '_lstm_cell_size', None)

trainer = None
# TODO: eager won't work b/c I removed ob_shape and ac_shape as arguments, but haven't done so for non-eager code
if args.alg == 'dqn':
    trainer = DQNTrainer(ob_shape, ac_shape, policy, discrete, gamma=args.gamma, target_update_freq=1000)
elif args.alg == 'pg':
    trainer = PGTrainer(ob_shape, ac_shape, policy, discrete, entcoeff=0)
elif args.alg == 'ppo':
    trainer = PPOTrainer(ob_shape, ac_shape, policy, discrete, epsilon=0.1, entcoeff=0)



runner = None
max_steps = float('inf') if args.max_episode_steps <= 0 else args.max_episode_steps
if args.alg == 'dqn':
    replay_buffer = ReplayBuffer(10 ** 6, 4)
    runner = DQNEnvironmentRunner(env, policy, replay_buffer, max_episode_steps=max_steps)
elif args.alg in ['pg', 'ppo']:
    runner = PGEnvironmentRunner(env, policy, args.gamma, max_episode_steps=max_steps)

benchmark_results['episode_rewards'] = []
agent_err = [0] * 100
value_err = [0] * 100
if not args.eager:
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

loglevels = {'debug' : logging.DEBUG, 'info' : logging.INFO, 'warning' : logging.WARNING}
logging.basicConfig(filename=None, level=loglevels[args.loglevel], format='%(message)s')

batch_size = 64 if args.alg == 'dqn' else 10

benchmark_file = None if args.benchmark is None else os.path.join('benchmark', args.benchmark) + '.pkl'

if args.alg == 'dqn':
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-3),
                                         (100000, 1e-4),
                                         (1000000,  5e-5),
                                    ],
                                    outside_value=5e-5)

    exploration = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (5e6, 0.01),
        ], outside_value=0.01
    )

    for t in itertools.count():
        if trainer.num_param_updates == args.max_learning_steps:
            from rl_algs.trainers.PPO import PPOTrainer
            break
        if benchmark_file and trainer.num_param_updates % 1000 == 0:
            with open(benchmark_file, 'wb') as f:
                pkl.dump(benchmark_results, f)
        done = runner.step(exploration.value(t))
        if done:
            logging.debug(runner.summary)
            benchmark_results['episode_rewards'].append(runner.episode_rew)
            runner.reset()

        if t > 50000 and t % 4 == 0 and replay_buffer.can_sample(batch_size):
            batch = replay_buffer.sample(batch_size)
            err = trainer.train(batch, lr_schedule.value(t))
            agent_err[trainer.num_param_updates % 100] = err

        if t % 10000 == 0:
            printstr = ''
            printstr += 'Timestep {:>6}, '.format(runner.episode_num)
            printstr += 'Param Updates {:>6}, '.format(trainer.num_param_updates)
            printstr += 'Reward {:>7.2f}, '.format(np.mean(benchmark_results['episode_rewards'][-100:]))
            printstr += 'Err {:>5.2f}, '.format(np.mean(agent_err[:trainer.num_param_updates]))
            printstr += 'Exploration {:.5f}'.format(exploration.value(t))
            logging.debug(printstr)

else:
    lr_mult = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-3*lr_mult),
                                         (300, 1e-4*lr_mult),
                                         (10000,  5e-5*lr_mult),
                                    ],
                                    outside_value=5e-5*lr_mult)

    for t in itertools.count():
        if trainer.num_param_updates == args.max_learning_steps:
            break
        if benchmark_file and trainer.num_param_updates % 1000 == 0:
            with open(benchmark_file, 'wb') as f:
                pkl.dump(benchmark_results, f)
        rollouts = []
        for _ in range(batch_size):
            runner.reset()
            policy.clear_memory()
            rollouts.append(runner.get_rollout())
            logging.debug(runner.summary)
            benchmark_results['episode_rewards'].append(runner.episode_rew)

        batch = {key : np.concatenate([rollout[key] for rollout in rollouts]) for key in rollouts[0]}
        if args.policy == 'lstm':
            batch['obs'] = batch['obs'].reshape((batch_size, -1) + ob_shape)

        policy.clear_memory()
        err, v_err = trainer.train(batch, lr_schedule.value(t))
        agent_err[trainer.num_param_updates % 100] = err
        value_err[trainer.num_param_updates % 100] = v_err

        if t % 3 == 0:
            printstr = ''
            printstr += 'Timestep {:>6}, '.format(runner.episode_num)
            printstr += 'Param Updates {:>6}, '.format(trainer.num_param_updates)
            printstr += 'Reward {:>7.2f}, '.format(np.mean(benchmark_results['episode_rewards'][-100:]))
            printstr += 'Err {:>5.2f}, '.format(np.mean(agent_err[:trainer.num_param_updates]))
            printstr += 'ValErr {:>5.2f}'.format(np.mean(value_err[:trainer.num_param_updates]))
            logging.info(printstr)
if benchmark_file:
    with open(benchmark_file, 'wb') as f:
        pkl.dump(benchmark_results, f)
