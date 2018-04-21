import os
import logging
import itertools

import numpy as np
import tensorflow as tf
import gym

from env_runners import DQNEnvironmentRunner
from policies.DQN import DQNAgent
from utils import ReplayBuffer, PiecewiseSchedule

BATCH_SIZE = 64
GAMMA = 0.99

LEARNING_STARTS = 50000
FRAME_HISTORY_LEN = 1
TARGET_UPDATE_FREQ = 10000
SAVE_FREQ = 10000
LEARN_FREQ = 4

LOG_EVERY_N_STEPS = 10000

def train_dqn(env, expname, logdir=None, sess=None):
    if sess is None:
        sess = tf.Session()

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    # Various setup stuff
    img_h, img_w, img_c = env.observation_space.shape
    input_shape = (img_h, img_w, FRAME_HISTORY_LEN * img_c)
    
    num_actions = env.action_space.n

    agent = DQNAgent(input_shape, num_actions, scope='agent')
    agent.setup_for_training(GAMMA)

    replay_buffer = ReplayBuffer(10 ** 5, FRAME_HISTORY_LEN)

    if logdir is not None:
        writer = tf.summary.FileWriter(os.path.join(logdir, 'results', expname))
        if not os.path.exists(os.path.join(logdir, 'models', expname)):
            os.mkdir(os.path.join(logdir, 'models', expname))

    logfile = None if logdir is None else os.path.join(logdir, 'episodes.log')
    # if logfile is not None:
    logging.basicConfig(filename=logfile, level=logging.DEBUG, filemode='w', format='%(message)s')

    runner = DQNEnvironmentRunner(env, agent, replay_buffer, 
                                    modifyreward=lambda rew : 1 if rew else -1, 
                                    verbose=True)


    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-3),
                                         (100000, 1e-3),
                                         (1000000,  5e-4),
                                    ],
                                    outside_value=5e-4)

    exploration = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (5e6, 0.01),
        ], outside_value=0.01
    )

    # Print Summaries
    num_param_updates = 0
    episode_rewards = [0] * 100
    episode_steps = [0] * 100
    agent_err = [0] * 100
    best_mean_episode_reward = -float('inf')

    # Tensor Summaries
    reward_summary = tf.Variable(0.)
    tf.summary.scalar("Episode_Reward", reward_summary)
    summary_ops = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    with sess.as_default():
        for t in itertools.count():
            done = runner.step(exploration.value(t))
            if done:
                if logdir is not None:
                    summary = sess.run(summary_ops, feed_dict={reward_summary : runner.episode_rew})
                    writer.add_summary(summary, runner.episode_num)
                    writer.flush()
                episode_rewards[runner.episode_num % 100] = runner.episode_rew
                episode_steps[runner.episode_num % 100] = runner.episode_steps
                runner.reset()

            if (t > LEARNING_STARTS and t % LEARN_FREQ == 0 and replay_buffer.can_sample(BATCH_SIZE)):
                batch = replay_buffer.sample(BATCH_SIZE)
                err = agent.train(batch, lr_schedule.value(t))
                agent_err[num_param_updates % 100] = err

                if num_param_updates % TARGET_UPDATE_FREQ == 0:
                    agent.update_target_network()

                if logdir is not None and num_param_updates % SAVE_FREQ == 0:
                    agent.save_model(os.path.join(logdir, 'models', expname, 'weights'))

                num_param_updates += 1

            best_mean_episode_reward = max(best_mean_episode_reward, np.mean(episode_rewards))
            if t % LOG_EVERY_N_STEPS == 0:
                printstr = '\n'
                printstr += 'Timestep {}, '.format(t)
                printstr += 'Param Updates {}, '.format(num_param_updates)
                printstr += 'Mean Reward {:.2f}, '.format(np.mean(episode_rewards))
                printstr += 'Mean Err {:.3f}, '.format(np.mean(agent_err))
                printstr += 'Mean Steps {}, '.format(np.mean(episode_steps))
                printstr += 'Exploration {:.5f}\n'.format(exploration.value(t))
                logging.info(printstr)


