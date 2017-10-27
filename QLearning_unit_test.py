import unittest
import lib.QLearning as ql
import gym

class MyTestCase(unittest.TestCase):
    def test_qlearning(self):
        env = gym.envs.make("MountainCar-v0")
        qlearning = ql.QLearning(env)
        qlearning.learn()
        qlearning.play()
        assert(True)


if __name__ == '__main__':
    unittest.main()
