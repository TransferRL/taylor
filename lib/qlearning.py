import gym
import itertools
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
from lib import plotting
# from gym import wrappers

if "./" not in sys.path:
    sys.path.append("./")

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler


class Estimator():
    """
    Value Function approximator.
    """

    def __init__(self, env, scaler, featurizer):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.scaler = scaler
        self.featurizer = featurizer
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)


    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])


class QLearning():

    def __init__(self, env):
        # Feature Preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(observation_examples)

        # Used to converte a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        featurizer = sklearn.pipeline.FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=100))
                ])
        featurizer.fit(scaler.transform(observation_examples))

        self.estimator = Estimator(env, scaler, featurizer)
        self.env = env


    def make_epsilon_greedy_policy(self, estimator, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

        Args:
            estimator: An estimator that returns q values for a given state
            epsilon: The probability to select a random action . float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.

        """

        def policy_fn(observation):
            A = np.ones(nA, dtype=float) * epsilon / nA
            q_values = estimator.predict(observation)
            best_action = np.argmax(q_values)
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fn


    def q_learning(self, env, estimator, num_episodes, discount_factor=1.0, epsilon=0.5, epsilon_decay=1.0):
        """
        Q-Learning algorithm for fff-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
            env: OpenAI environment.
            estimator: Action-Value function estimator
            num_episodes: Number of episodes to run for.
            discount_factor: Lambda time discount factor.
            epsilon: Chance the sample a random action. Float betwen 0 and 1.
            epsilon_decay: Each episode, epsilon is decayed by this factor

        Returns:
            An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """

        # Keeps track of useful statistics
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

        for i_episode in range(num_episodes):

            # The policy we're following
            policy = self.make_epsilon_greedy_policy(
                estimator, epsilon * epsilon_decay ** i_episode, env.action_space.n)

            # Print out which episode we're on, useful for debugging.
            # Also print reward for last episode
            last_reward = stats.episode_rewards[i_episode - 1]
            sys.stdout.flush()

            # Reset the environment and pick the first action
            state = env.reset()

            # Only used for SARSA, not Q-Learning
            next_action = None

            # One step in the environment
            for t in itertools.count():

                # Choose an action to take
                # If we're using SARSA we already decided in the previous step
                if next_action is None:
                    action_probs = policy(state)
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                else:
                    action = next_action

                # Take a step
                next_state, reward, done, _ = env.step(action)

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                # TD Update
                q_values_next = estimator.predict(next_state)

                # Use this code for Q-Learning
                # Q-Value TD Target
                td_target = reward + discount_factor * np.max(q_values_next)

                # Use this code for SARSA TD Target for on policy-training:
                # next_action_probs = policy(next_state)
                # next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
                # td_target = reward + discount_factor * q_values_next[next_action]

                # Update the function approximator using our target
                estimator.update(state, action, td_target)

                print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")

                if done:
                    break

                state = next_state



    def learn(self):
        self.q_learning(self.env, self.estimator, 100, epsilon=0.0)


    def play(self):
        done = 0
        policy = self.make_epsilon_greedy_policy(self.estimator, 0, self.env.action_space.n)
        state = self.env.reset()
        for i_episode in range(100000):
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
                print('done: {}'.format(state))
            #     plt.figure()
            #     plt.imshow(env.render(mode='rgb_array'))
                # break
            state = next_state





