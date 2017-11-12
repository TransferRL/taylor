import unittest
import lib.qlearning as ql
import gym
import lib.env.mountain_car

class MyTestCase(unittest.TestCase):
    def test_qlearning(self):
        # env = gym.envs.make("MountainCar-v0")
        env = lib.env.mountain_car.MountainCarEnv()
        qlearning = ql.QLearning(env, rendering=True)
        qlearning.learn()
        dsource = qlearning.play()
        assert(True)


if __name__ == '__main__':
    unittest.main()
