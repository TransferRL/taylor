from __future__ import print_function

import numpy as np
import tensorflow as tf
from lib.env.threedmountain_car import ThreeDMountainCarEnv
import lib.qlearning as ql
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


def model_fn(features, labels, mode):

    hidden_layer_1 = tf.layers.dense(features["x"], n_hidden_1)
    hidden_layer_2 = tf.layers.dense(hidden_layer_1, n_hidden_2)
    output = tf.layers.dense(hidden_layer_2, num_output)

    # output = tf.reshape(out_layer, [-1])
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions={"output", output}
        )

    # print(labels.shape)
    # print(output.shape)

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
        # print(f_read['dsource'].shape)
        dsource = f_read['dsource']
    else:
        qlearning_2d = ql.QLearning(mc2d_env)
        qlearning_2d.learn()
        dsource = np.array(qlearning_2d.play())
        # print(dsource.shape)
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

    dtarget_x = np.array([np.append(x[0], x[1]) for x in dtarget])
    dtarget_y = np.array([x[2] for x in dtarget])

    dtarget_train_x = dtarget_x[:-100]
    dtarget_train_y = dtarget_y[:-100]
    dtarget_test_x = dtarget_x[-100:]
    dtarget_test_y = dtarget_y[-100:]



    # train one step transition model
    nn = tf.estimator.Estimator(model_fn=model_fn)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": dtarget_train_x},
        y=dtarget_train_y,
        num_epochs=None,
        shuffle=True
    )

    # print(dtarget_train_x.shape)
    # print(dtarget_train_y.shape)

    nn.train(input_fn=train_input_fn, steps=num_steps)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": dtarget_test_x},
        y=dtarget_test_y,
        num_epochs=1,
        shuffle=False
    )


    ev = nn.evaluate(input_fn=test_input_fn)

    print("loss: %s" % ev["loss"])
    print("Root Mean Squared Error: %s" % ev["rmse"])


    # Find the mapping between source and target
    mc2d_states = mc2d_env.observation_space.shape[0] # 2
    mc3d_states = mc3d_env.observation_space.shape[0] # 4
    mc2d_actions = mc2d_env.action_space.n # 3
    mc3d_actions = mc3d_env.action_space.n # 5

    mse_state_mappings = np.zeros((2,)*mc3d_states) # 2 by 2 by 2 by 2
    mse_action_mappings = np.ndarray(shape=(mc3d_actions,mc2d_actions, mc3d_states*mc3d_states)) # 5 by 3 by 16
    mse_action_mappings.fill(-1)

    state_count = 0
    for s0 in range(mc2d_states): # s0 is the first state of target states, x
        for s1 in range(mc2d_states): # s1 is second state of target states, y
            for s2 in range(mc2d_states):  # s2 is third state of target states, x_dot
                for s3 in range(mc2d_states):  # s3 is fourth state of target states, y_dot

                    state_losses = []

                    for a_mc3d in range(mc3d_actions):
                        for a_mc2d in range(mc2d_actions):
                            states = np.array([x[0] for x in dsource if x[1]==a_mc2d])
                            actions = np.array([x[1] for x in dsource if x[1] == a_mc2d])
                            n_states = np.array([x[2] for x in dsource if x[1]==a_mc2d])

                            if (states.size==0) or (actions.size==0) or (n_states.size==0):
                                print('this happened..') # TODO
                                # mse_action_mappings[a_mc3d, a_mc2d, state_count] = 0
                                continue

                            # transform to dsource_trans
                            actions_trans = np.ndarray(shape=(actions.size,))
                            actions_trans.fill(a_mc3d)
                            input_trans = np.array([states[:, s0], states[:, s1], states[:, s2], states[:, s3], actions_trans]).T
                            # input_trans = [states_trans, actions]
                            n_states_trans = np.array([n_states[:,s0], n_states[:,s1], n_states[:,s2], n_states[:,s3]]).T

                            # calculate mapping error
                            test_input_fn = tf.estimator.inputs.numpy_input_fn(
                                x={"x": input_trans},
                                y=n_states_trans,
                                num_epochs=1,
                                shuffle=False
                            )
                            ev = nn.evaluate(input_fn=test_input_fn)
                            # loss_mapping = sess.run(loss_op, feed_dict={X: input_trans, Y: n_states_trans})
                            # print('loss_mapping is {}'.format(loss_mapping))

                            state_losses.append(ev["loss"])
                            mse_action_mappings[a_mc3d, a_mc2d, state_count] = ev["loss"]

                    mse_state_mappings[s0, s1, s2, s3] = np.mean(state_losses)
                    state_count += 1

    # mse_action_mappings_result = [[np.mean(mse_action_mappings[a_mc3d, a_mc2d, :]) for a_mc2d in range(mc2d_actions)] for a_mc3d in range(mc3d_actions)]

    mse_action_mappings_result = np.zeros((mc3d_actions, mc2d_actions))
    for a_mc3d in range(mc3d_actions):
        for a_mc2d in range(mc2d_actions):
            losses_act = []
            for s in range(mc3d_states*mc3d_states):
                if mse_action_mappings[a_mc3d, a_mc2d, s] != -1:
                    # print(mse_action_mappings[a_mc3d, a_mc2d, s])
                    losses_act.append(mse_action_mappings[a_mc3d, a_mc2d, s])
            mse_action_mappings_result[a_mc3d, a_mc2d] = np.mean(losses_act)


    print('action mapping: {}'.format(mse_action_mappings_result))
    print('state mapping {}'.format(mse_state_mappings))


    print('x,x,x,x: {}'.format(mse_state_mappings[0][0][0][0]))
    print('x,x,x,x_dot: {}'.format(mse_state_mappings[0][0][0][1]))
    print('x,x,x_dot,x: {}'.format(mse_state_mappings[0][0][1][0]))
    print('x,x,x_dot,x_dot: {}'.format(mse_state_mappings[0][0][1][1]))
    print('x,x_dot,x,x: {}'.format(mse_state_mappings[0][1][0][0]))
    print('x,x_dot,x,x_dot: {}'.format(mse_state_mappings[0][1][0][1]))
    print('x,x_dot,x_dot,x: {}'.format(mse_state_mappings[0][1][1][0]))
    print('x,x_dot,x_dot,x_dot: {}'.format(mse_state_mappings[0][1][1][1]))
    print('x_dot,x,x,x: {}'.format(mse_state_mappings[1][0][0][0]))
    print('x_dot,x,x,x_dot: {}'.format(mse_state_mappings[1][0][1][0]))
    print('x_dot,x,x_dot,x: {}'.format(mse_state_mappings[1][0][1][1]))
    print('x_dot,x,x_dot,x_dot: {}'.format(mse_state_mappings[1][1][0][0]))
    print('x_dot,x_dot,x,x: {}'.format(mse_state_mappings[1][0][0][1]))
    print('x_dot,x_dot,x,x_dot: {}'.format(mse_state_mappings[1][1][0][1]))
    print('x_dot,x_dot,x_dot,x: {}'.format(mse_state_mappings[1][1][1][0]))
    print('x_dot,x_dot,x_dot,x_dot: {}'.format(mse_state_mappings[1][1][1][1]))






if __name__ == '__main__':
    main()