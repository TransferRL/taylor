import gym
import itertools
import matplotlib
import numpy as np
import tensorflow as tf
from lib.env.threedmountain_car import ThreeDMountainCarEnv
import lib.QLearning as ql
import lib.RandomAction
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.1
num_steps = 10000
batch_size = 100
display_step = 100

# Network Parameters
n_hidden_1 = 32 # 1st layer number of neurons
n_hidden_2 = 32 # 2nd layer number of neurons
num_input = 5
num_output = 4

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_output])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_output]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.abs(logits - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(logits), tf.round(Y)), tf.int32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()



# get envs
mc2d_env = gym.envs.make("MountainCar-v0")
mc3d_env = ThreeDMountainCarEnv()

# train source task
# qlearning_2d = ql.QLearning(mc2d_env)
# qlearning_2d.learn()
# dsource = qlearning_2d.play()
# print(dsource)

# do random action for target task
random_action_3d = lib.RandomAction.RandomAction(mc3d_env)
dtarget = random_action_3d.play()
# print(dtarget)

# approximate the one-step transition model
# Define the input function for training
dsa = np.array([np.append(x[0], x[1]) for x in dtarget])
dns = np.array([x[2] for x in dtarget])

# print(dsa.shape)
# print(dsa)

dsa_train = dsa[:-100]
dns_train = dns[:-100]
dsa_test = dsa[-100:]
dns_test = dns[-100:]

# print(dsa_train.shape)

batch_num = np.size(dsa_train, 0)
# print(batch_num)

loss = []
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(num_steps):
        #         batch_x, batch_y = mnist.train.next_batch(batch_size)
        # batch_x = np.zeros((batch_size, 5))
        # batch_y = np.zeros((batch_size, 4))
        batch_x = dsa_train[(step*batch_size)%batch_num : (step*batch_size+batch_size)%batch_num, : ]
        batch_y = dns_train[(step*batch_size)%batch_num : (step*batch_size+batch_size)%batch_num, : ]
        # print((step*batch_size)%batch_num)
        # print((step*batch_size+batch_size-1)%batch_num)
        # print(batch_x.shape)
        # print(batch_y.shape)

        # Run optimization op (backprop)
        loss_train, _ = sess.run([loss_op, train_op], feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0:
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss_train) )
            loss.append(loss_train)

    print("Optimization Finished!")
    # plot training loss
    plt.plot(loss)
    plt.show()

    # test set
    loss_test = sess.run(loss_op, feed_dict={X: dsa_test, Y: dns_test})
    print("test loss is {}".format(loss_test))











