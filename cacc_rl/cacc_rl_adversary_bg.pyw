

from random import seed
from random import randint
import numpy as np
import matplotlib.pyplot as plt

#keras (not needed)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

#gym
import gym
from gym_cacc import Vehicle
from gym_cacc.envs import Adversary

#models
from models.dqn_model1 import DQNAgent


#hyper parameters
EPISODES	= 10000
STEP_LIMIT  = 500 #steps
BATCH_SIZE	= 32

#agent parameters
GAMMA_REAR			= 0.95
EPSILON_REAR		= 1.0
EPSILON_MIN_REAR	= 0.01
EPSILON_DECAY_REAR	= 0.99
LEARNING_RATE_REAR	= 0.001
MEMORY_SIZE_REAR	= 2000

#agent parameters
GAMMA_FRONT			= 0.95
EPSILON_FRONT		= 1.0
EPSILON_MIN_FRONT	= 0.01
EPSILON_DECAY_FRONT	= 0.99
LEARNING_RATE_FRONT	= 0.001
MEMORY_SIZE_FRONT	= 2000

#vehicle parameters
#stats	:= (length, width, height, weigth, top_vel, top_acc, top_jerk)
#		:= (m, m, m, kg, m/s, m/s^2, m/s^3)
VEHICLE_STATISTICS = (4.67312, 1.81102, 1.43002, 1617.51, 69.2912, 5.15815, 1)
VEHICLE = Vehicle(statistics = VEHICLE_STATISTICS)

#environment parameters
TARGET_HEADWAY = 2
REACTION_TIME_LOWER_BOUND = 10 #frames (steps)
REACTION_TIME_UPPER_BOUND = 15 #frames (steps)



#print('starting main code')

if __name__ == '__main__':
	#create environment and agent
	env = gym.make('cacc-adversary-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	#setting environment variables and agent hyperparameters
	env.reset(VEHICLE, TARGET_HEADWAY)
	
	agent_front  = DQNAgent(state_size, action_size, GAMMA_FRONT, EPSILON_FRONT, EPSILON_MIN_FRONT, EPSILON_DECAY_FRONT, LEARNING_RATE_FRONT, MEMORY_SIZE_FRONT)
	agent_rear   = DQNAgent(state_size, action_size, GAMMA_REAR,  EPSILON_REAR,  EPSILON_MIN_REAR,  EPSILON_DECAY_REAR,  LEARNING_RATE_REAR,  MEMORY_SIZE_REAR)

	# agent.load("./save/cartpole-ddqn.h5")
	done = False


	environment_state_memory = []


	#print('begin training')
	#print('-------------------------------')

	for e in range(EPISODES):

		environment_state_memory.append([])


		#reset environment and get initial state
		state = env.reset()
		state_front, state_rear = state

		state_front = np.reshape(state_front, [1, state_size])
		state_rear  = np.reshape(state_rear,  [1, state_size])

		episode_reward_front = 0
		episode_reward_rear  = 0

		action_delay_front = randint(10, 15)
		action_delay_rear  = randint(10, 15)

		action_delay_timer_front = action_delay_front #allow agents to input on first step
		action_delay_timer_rear  = action_delay_rear


		for time in range(STEP_LIMIT):

			environment_state_memory[e].append((env.variablesKinematicsFrontVehicle(), 
												env.variablesKinematicsRearVehicle(), 
												env.variablesEnvironment(), 
												[episode_reward_front, episode_reward_rear]))

			#if time % 10 == 0:
			#if True:
				#print('-----------')
				#print('current env: front vehicle :', env.variablesKinematicsFrontVehicle(), '[',time + 1, '/', e + 1,']')
				#print('             rear  vehicle :', env.variablesKinematicsRearVehicle())
				#print('             other vars    :', env.variablesEnvironment())
				#print('             EPISODE_REWARD: (' + str(episode_reward_front) +' ' + str(episode_reward_rear) + ')')

			#agents act on environment's current state
			#front agent
			if action_delay_timer_front == action_delay_front:
				action_delay_timer_front = 0
				action_delay_front = randint(10, 15)
				action_front = agent_front.act(state_front)
			
			#rear agent
			if action_delay_timer_rear == action_delay_rear:
				action_delay_timer_rear = 0
				action_delay_rear = randint(10, 15)
				action_rear = agent_rear.act(state_rear)

			action = (action_front, action_rear)

			action_delay_timer_front += 1
			action_delay_timer_rear  += 1


			#environment step
			next_states, rewards, done, _ = env.step(action)
			#next_states, rewards, done, _ = env.step((20,20))


			next_state_front, next_state_rear = next_states
			reward_front, reward_rear = rewards


			#update the epside reward
			episode_reward_front += reward_front
			episode_reward_rear  += reward_rear

			#remember
			#front vehicle
			next_state_front = np.reshape(next_state_front, [1, state_size])
			agent_front.remember(state_front, action_front, reward_front, next_state_front, done)
			state_front = next_state_front

			#rear vehicle
			next_state_rear = np.reshape(next_state_rear, [1, state_size])
			agent_rear.remember(state_rear, action_rear, reward_rear, next_state_rear, done)
			state_rear = next_state_rear

			if done:
				agent_front.update_target_model()
				agent_rear.update_target_model()
				#print('-----------')
				#print('episode: {0:3d}/{1:3d} :: FRONT: score: {2}, e: {3:.2}'
				#	  .format(e, EPISODES, episode_reward_front, agent_front.epsilon))
				#print('                 :: REAR:  score: {0}, e: {1:.2}'
				#	  .format(episode_reward_rear, agent_rear.epsilon))
				break

			if len(agent_front.memory) > BATCH_SIZE:
				agent_front.replay(BATCH_SIZE)

			if len(agent_rear.memory) > BATCH_SIZE:
				agent_rear.replay(BATCH_SIZE)


		#saving agents
		if e % 10 == 0:
			agent_front.save('cacc_rl_adversary_front_'+str(e)+'.h5')
			agent_rear.save('cacc_rl_adversary_rear_'+str(e)+'.h5')

		#if e % 10 == 0:
		#	agent.save("./save/cartpole-ddqn.h5")
		

		#print(environment_state_memory[0])

		#print('\n---------')
		#print('DONE!!!')
		#print('---------\n')


		save_to_file = False

		if save_to_file:
			print('saving env state memory to file')
			import pickle

			with open('env_state_mem.txt', 'wb') as fp:
				pickle.dump(environment_state_memory, fp)

		



