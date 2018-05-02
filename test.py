import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from rl_algs.train_pg import train_pg
import gym
import gym_gridworld

# You won't actually be able to run this because it requires my gridworld
# It's just an example to show all you should need to do to run the training

env = gym.make('gridworld-v0')
env.load('/home/rmrao/projects/curriculum/worlds/doors.pkl')
train_pg(env, 'test')
