"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

'''
    Jeremy:
        This is a modified environment for 3D mountain car.
        Currently the rendering only shows the x and z axises 
        (i.e. The 3D graphics is projected onto the (0,1,0) plane) 

        More to come..
'''

class ThreeDMountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_position_x = -1.2
        self.max_position_x = 0.6
        self.max_speed_x = 0.07

        # jm: The second coordinate bounds
        self.min_position_y = -1.2
        self.max_position_y = 0.6
        self.max_speed_y = 0.07

        self.goal_position = 0.5

        self.low = np.array([self.min_position_x, self.min_position_y, -self.max_speed_x, -self.max_speed_y]) # jm: x,y,x_dot,y_dot
        self.high = np.array([self.max_position_x, self.max_position_y, self.max_speed_x, self.max_speed_y]) # jm: x,y,x_dot,y_dot

        self.viewer = None

        self.action_space = spaces.Discrete(5) # jm: {Neutral, West, East, South, North}
        self.observation_space = spaces.Box(self.low, self.high)

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position_x, position_y, velocity_x, velocity_y = self.state
        if action == 0: # neutral
            velocity_x += math.cos(3*position_x)*(-0.0025)
            velocity_y += math.cos(3*position_y)*(-0.0025)
        elif action == 1: # left x, west
            velocity_x += math.cos(3*position_x)*(-0.0025) - 0.001
            velocity_y += math.cos(3*position_y)*(-0.0025)
        elif action == 2: # right x, east
            velocity_x += math.cos(3*position_x)*(-0.0025) + 0.001
            velocity_y += math.cos(3*position_y)*(-0.0025)
        elif action == 3: # left y, south
            velocity_x += math.cos(3*position_x)*(-0.0025)
            velocity_y += math.cos(3*position_y)*(-0.0025) - 0.001
        elif action == 4:
            velocity_x += math.cos(3*position_x)*(-0.0025)
            velocity_y += math.cos(3*position_y)*(-0.0025) + 0.001


        velocity_x = np.clip(velocity_x, -self.max_speed_x, self.max_speed_x)
        velocity_y = np.clip(velocity_y, -self.max_speed_y, self.max_speed_y)

        position_x += velocity_x
        position_y += velocity_y

        position_x = np.clip(position_x, self.min_position_x, self.max_position_x)
        position_y = np.clip(position_y, self.min_position_y, self.max_position_y)

        if (position_x == self.min_position_x and velocity_x<0):
            velocity_x = 0

        if (position_y == self.min_position_y and velocity_y<0):
            velocity_y = 0

        done = bool(position_x >= self.goal_position and position_y >= self.goal_position)
        # done = bool(position_x >= self.goal_position)

        reward = -1.0

        self.state = (position_x, position_y, velocity_x, velocity_y)
        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), self.np_random.uniform(low=-0.6, high=-0.4), 0, 0])
        return np.array(self.state)

    # def _height(self, xs, ys):
    #     pos = np.sqrt(np.square(xs) + np.square(ys))
    #     return np.sin(3 * pos)*.45+.55

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55


    def _render(self, mode='human', close=False):
        # jm: only showing x positions for now.. More too add..
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_position_x - self.min_position_x
        scale = screen_width/world_width
        carwidth=40
        carheight=20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position_x, self.max_position_x, 100)
            # ys = np.linspace(self.min_position_y, self.max_position_y, 100)
            # ys = np.zeros(100)
            zs = self._height(xs)
            xyzs = list(zip((xs-self.min_position_x)*scale, zs*scale))

            self.track = rendering.make_polyline(xyzs)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position_x)*scale
            flagy1 = self._height(self.goal_position)*scale #jm: need to change this
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position_x)*scale, self._height(pos)*scale) #jm: need to change this
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close_gui(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return