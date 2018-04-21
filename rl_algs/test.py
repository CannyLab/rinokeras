from dqn import train_dqn
import gym
import gym_gridworld

# You won't actually be able to run this because it requires my gridworld
# It's just an example to show all you should need to do to run the training

env = gym.make('gridworld-v0')
env.load('../pong-teacher/worlds/simple.npy')
train_dqn(env, 'test')
