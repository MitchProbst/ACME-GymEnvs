"""
Classic cart-pole system with continuous action implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import logging
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class CartPoleContinuousEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        # For some reason in Gym they define this length to be half of the pole's length
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)
        self.max_force = 2000.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode (Irrelevant for Optimal Control)
        # Note: Changing the size of x_threshold will dynamically change the window-size as well. Smaller values make the cart and pole appear bigger.
        self.theta_threshold_radians = 360 * 2 * np.pi / 360
        self.x_threshold = 50
        
        
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        # Defines what actions are possible
        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def state_eq(self, st, u):
        x, x_dot, theta, theta_dot = st
        force = u[0]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # The order in which the velocity/position updated were switched according to this issue:
        # https://github.com/openai/gym/issues/907
        # See original CartPole in Gym for traditional order
        x_dot = x_dot + self.tau * xacc
        x  = x + self.tau * x_dot
        theta_dot = theta_dot + self.tau * thetaacc
        theta = theta + self.tau * theta_dot
        
        return np.array([x, x_dot, theta, theta_dot])

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        self.state = self.state_eq(state, action) 
        x, x_dot, theta, theta_dot = self.state
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return self.state, reward, done, {}

    def reset(self):
        # Instantiate the cart with random, reasonable values
        x = np.random.uniform(low=-3,high=3)
        x_dot = np.random.uniform(low=-3,high=3)
        theta = np.random.uniform(low=-np.pi/3, high=np.pi/3)
        theta_dot = np.random.uniform(low=-np.pi/6,high=np.pi/6)
        self.state = [x, x_dot, theta, theta_dot]
        
        # For testing you may want to hard-code some kind of default state
        # If so, simply uncomment this line and change the values as you see fit.
        # self.state = [0, 0, 0.25, 0]
        
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        # Mess with this function at your own risk.
        screen_width = 1000
        world_width = self.x_threshold/4
        scale = screen_width/world_width
        
        carty = scale / 1.25 # TOP OF CART
        polewidth = scale / 12
        polelen = scale
        cartwidth = scale / 2.5
        cartheight = scale / 4
        
        screen_height = round(carty + polewidth + 75)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
        
    def close(self):
        if self.viewer: self.viewer.close()
        
class StochasticCartPoleContinuousEnv(CartPoleContinuousEnv):
    """
    A stochastic version where the mass is randomized at the beginning of each environment
    """
    def __init__(self):
        super().__init__()
        # This may be a bit too random, to be determined
        self.masscart = np.random.uniform(low=0.5, high=2.5)
        
        # Update total_mass because it depends on masscart
        self.total_mass = (self.masspole + self.masscart)