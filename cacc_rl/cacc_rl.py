
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
EPISODES	= 100
BATCH_SIZE	= 32

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
STEP_LIMIT = 500 #steps

print('starting main code')

if __name__ == '__main__':
	#create environment and agent
	env = gym.make('cacc-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	#setting environment variables and agent hyperparameters
	env.reset(VEHICLE, TARGET_HEADWAY)
	
	agent = DQNAgent(state_size, action_size, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY, LEARNING_RATE, MEMORY_SIZE)

	# agent.load("./save/cartpole-ddqn.h5")
	done = False

	print('begin training')
	print('-------------------------------')

	for e in range(EPISODES):
		#reset environment and get initial state
		state = env.reset()
		state = np.reshape(state, [1, state_size])
		episode_reward = 0

		for time in range(STEP_LIMIT):
			if time % 15 == 0:
				print('current env: front vehicle :', env.variablesKinematicsFrontVehicle(), '[',time,']')
				print('             rear  vehicle :', env.variablesKinematicsRearVehicle())
				print('             other vars    :', env.variablesEnvironment())
				print('             EPISODE_REWARD:', episode_reward)

			#agent acts on environment's current state
			action = agent.act(state)
			next_state, reward, done, _ = env.step(action)

			#update the epside reward
			episode_reward += reward

			#remember
			next_state = np.reshape(next_state, [1, state_size])
			agent.remember(state, action, reward, next_state, done)
			state = next_state

			if done:
				agent.update_target_model()
				print('episode: {}/{}, score: {}, e: {:.2}'
					  .format(e, EPISODES, episode_reward, agent.epsilon))
				break

			if len(agent.memory) > BATCH_SIZE:
				agent.replay(BATCH_SIZE)


		#if e % 10 == 0:
		#	agent.save("./save/cartpole-ddqn.h5")





