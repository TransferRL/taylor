import gym
import numpy as np
from matplotlib import pyplot as plt
import itertools
from lib.env.mountain_car import MountainCarEnv

env = MountainCarEnv()

state = env.reset()

for t in itertools.count():
	action = env.action_space.sample()
	state, reward, done, info = env.step(action)
	env.render()



