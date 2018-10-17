import gym
from gym import error, spaces, utils
from gym.utils import seeding

class CarEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self):
        print('creating env')
        pass

    def step(self, action):
        print('stepping env')
        pass

    def reset(self):
        print('reseting env')
        pass

    def render(self, mode='human', close=False):
        print('rendering env')
        pass
