import gym
import itertools
import matplotlib
import numpy as np
from lib.env.threedmountain_car import ThreeDMountainCarEnv
import lib.QLearning as ql
import lib.RandomAction


mc2d_env = gym.envs.make("MountainCar-v0")
mc3d_env = ThreeDMountainCarEnv()

# train source task

qlearning_2d = ql.QLearning(mc2d_env)
qlearning_2d.learn()
dsource = qlearning_2d.play()

# print(dsource)

# do random action for target task
random_action_3d = lib.RandomAction.RandomAction(mc3d_env)
dtarget = random_action_3d.play()

# print(dtarget)

# approximate the one-step transition model








