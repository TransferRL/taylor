from __future__ import print_function

import numpy as np
import tensorflow as tf
from lib.env.threedmountain_car import ThreeDMountainCarEnv
import lib.QLearning as ql
import lib.env.mountain_car
import os
import lib.RandomAction


def main():
    # get envs
    mc2d_env = lib.env.mountain_car.MountainCarEnv()
    mc3d_env = ThreeDMountainCarEnv()

    # source task
    if os.path.isfile('./dsource_qlearn.npz'):
        f_read = np.load('./dsource_qlearn.npz')
        print(f_read['dsource'].shape)
    else:
        qlearning_2d = ql.QLearning(mc2d_env)
        qlearning_2d.learn()
        dsource = np.array(qlearning_2d.play())
        print(dsource.shape)
        np.savez('dsource_qlearn.npz', dsource = dsource)

    # target task
    if os.path.isfile('./dtarget_random.npz'):
        f_read = np.load('./dtarget_random.npz')
        # print(f_read['dtarget'].shape)
        dtarget = f_read['dtarget']
    else:
        random_action_3d = lib.RandomAction.RandomAction(mc3d_env)
        dtarget = np.array(random_action_3d.play())
        np.savez('./dtarget_random.npz', dtarget = dtarget)







if __name__ == '__main__':
    main()