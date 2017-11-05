from lib.env.threedmountain_car import ThreeDMountainCarEnv
from lib.env.mountain_car import MountainCarEnv
import math
import numpy as np

class StateSampler():

    def __init__(self, seed=None):
        self.seed = seed

    def getRandom2DState(self,with_velocity=True):
        env = MountainCarEnv()
        env.reset()

        random_pos = np.random.uniform(low=env.min_position, high=env.max_position)

        # TODO: calculates the maximum speed at this position
        random_velocity = np.random.uniform(low=-env.max_speed, high=env.max_speed) if with_velocity else 0
        return [random_pos, random_velocity]


    def getRandom3DState(self, with_velocity=True):
        env = ThreeDMountainCarEnv()
        env.reset()

        random_pos_x = np.random.uniform(low=env.min_position_x, high=env.max_position_y)
        random_pos_y = np.random.uniform(low=env.min_position_x, high=env.max_position_y)

        # TODO: calculates the maximum speed at this position
        random_velocity_x = np.random.uniform(low=-env.max_speed_x, high=env.max_speed_x) if with_velocity else 0
        random_velocity_y = np.random.uniform(low=-env.max_speed_y, high=env.max_speed_y) if with_velocity else 0


        return [random_pos_x, random_pos_y, random_velocity_x, random_velocity_y]