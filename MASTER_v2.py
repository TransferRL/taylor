from __future__ import print_function

import numpy as np
import tensorflow as tf
from lib.env.threedmountain_car import ThreeDMountainCarEnv
import lib.QLearning as ql
import lib.env.mountain_car
import os
import lib.RandomAction

learning_rate = 0.1
num_steps = 10000
batch_size = 100
display_step = 100

n_hidden_1 = 32 # 1st layer number of neurons
n_hidden_2 = 32 # 2nd layer number of neurons
num_input = 5
num_output = 4


def model_fn(features, labels, mode, params):

    hidden_layer_1 = tf.layers.dense(features["x"], n_hidden_1)
    hidden_layer_2 = tf.layers.dense(hidden_layer_1, n_hidden_2)
    out_layer = tf.layers.dense(hidden_layer_2, num_output)

    output = tf.reshape(out_layer, [-1])
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions={"output", output}
        )

    loss = tf.losses.mean_squared_error(labels, output)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    eval_metric_ops={
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float64), output
        )
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )



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


    # train one step transition model
    nn = tf.estimator.Estimator(model_fn=model_fn)

    




if __name__ == '__main__':
    main()