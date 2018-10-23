
import random
import numpy as np

#keras (not needed)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

#gym
import gym
from gym_cacc import Vehicle
from gym_cacc.envs import StopAndGo

#models
from models.dqn_model1 import DQNAgent


#hyper parameters
EPISODES = 100

#agent parameters
GAMMA			= 0.95
EPSILON			= 1.0
EPSILON_MIN		= 0.01
EPSILON_DECAY	= 0.99
LEARNING_RATE	= 0.001
MEMORY_SIZE		= 2000

#vehicle parameters
#stats := (length, width, height, weigth, top_vel, top_acc, top_jerk)
VEHICLE_STATISTICS = (1,1,1,1,1,1,1)
VEHICLE = Vehicle(statistics = VEHICLE_STATISTICS)

#environment parameters
TARGET_HEADWAY = 2
TIME_LIMIT = 15 #seconds


# NEED TO LOOK AT!!!
if __name__ == '__main__':
    env = gym.make('cacc-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-ddqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")





