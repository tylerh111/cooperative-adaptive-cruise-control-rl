
from random import seed
from random import randint
import numpy as np
import time
import threading

import pickle


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
from models.dqn_model import DQNAgent


#project path
PROJECT_PATH = 'E:\\comp594\\Cooperative-Adaptive-CC-RL\\cacc_rl\\'


#hyper parameters
EPISODES	= 10005
STEP_LIMIT  = 2048 #steps
BATCH_SIZE	= 32

#agent hyper parameters
GAMMA_FRONT			= 0.95
EPSILON_FRONT		= 1.0
EPSILON_MIN_FRONT	= 0.01
EPSILON_DECAY_FRONT	= 0.99
LEARNING_RATE_FRONT	= 0.000001
MEMORY_SIZE_FRONT	= 16384

#agent hyper parameters
GAMMA_REAR			= 0.95
EPSILON_REAR		= 1.0
EPSILON_MIN_REAR	= 0.01
EPSILON_DECAY_REAR	= 0.99
LEARNING_RATE_REAR	= 0.000001
MEMORY_SIZE_REAR	= 16384

#vehicle parameters
#stats	:= (length, width, height, weigth, top_vel, top_acc, top_jerk)
#		:= (m, m, m, kg, m/s, m/s^2, m/s^3)
VEHICLE_STATISTICS = (4.67312, 1.81102, 1.43002, 1617.51, 55, 3, 12)
VEHICLE = Vehicle(statistics = VEHICLE_STATISTICS)

#environment parameters
TARGET_HEADWAY = 2
GRANULARITY = 21

INIT_VELOCITY = 25
INIT_ACCELERATION = 0
ALPHA_VELOCITY = 4
ALPHA_ACCELERATION = 2

REACTION_TIME_LOWER_BOUND = 10 #frames (steps)
REACTION_TIME_UPPER_BOUND = 15 #frames (steps)


#names
VERSION = 'a_1_3'
WEIGHT_NAME = 'cacc_rl_adversary_'+VERSION
PATH_TO_WEIGHTS = PROJECT_PATH+'weights\\dqn_model_adversary\\'



print('starting main code')

if __name__ == '__main__':
	#create environment and agent
	env = gym.make('cacc-adversary-v0')

	#setting environment variables and agent hyperparameters
	env.reset(VEHICLE, TARGET_HEADWAY, GRANULARITY, INIT_VELOCITY, ALPHA_VELOCITY, INIT_ACCELERATION, ALPHA_ACCELERATION)
	
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n


	#weights_path_front = PATH_TO_WEIGHTS+'front\\cacc_rl_adversary_1_1_front_50.h5'
	#weights_path_rear  = PATH_TO_WEIGHTS+'rear\\cacc_rl_adversary_1_1_rear_50.h5'
	weights_path_front = None
	weights_path_rear  = None

	agent_front  = DQNAgent(state_size, action_size, weights_path_front, GAMMA_FRONT, EPSILON_FRONT, EPSILON_MIN_FRONT, EPSILON_DECAY_FRONT, LEARNING_RATE_FRONT, MEMORY_SIZE_FRONT)
	agent_rear   = DQNAgent(state_size, action_size, weights_path_rear,  GAMMA_REAR,  EPSILON_REAR,  EPSILON_MIN_REAR,  EPSILON_DECAY_REAR,  LEARNING_RATE_REAR,  MEMORY_SIZE_REAR)

	print('weights loaded')

	print('begin training')
	print('-------------------------------')
	time.sleep(2)
	
	done = False

	environment_state_memory = []
	
	for e in range(EPISODES):

		environment_state_memory.append([])


		#reset environment and get initial state
		state = env.reset(VEHICLE, TARGET_HEADWAY, GRANULARITY, INIT_VELOCITY, ALPHA_VELOCITY, INIT_ACCELERATION, ALPHA_ACCELERATION)
		state_front, state_rear = state

		state_front = np.reshape(state_front, [1, state_size])
		state_rear  = np.reshape(state_rear,  [1, state_size])

		episode_reward_front = 0
		episode_reward_rear  = 0

		action_front = 0
		action_rear  = 0
		action = (action_front, action_rear)

		action_delay_front = randint(10, 15)
		action_delay_rear  = randint(10, 15)

		action_delay_timer_front = action_delay_front #allow agents to input on first step
		action_delay_timer_rear  = action_delay_rear


		for t in range(STEP_LIMIT):
			
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
			next_states, rewards, done, extra = env.step(action)
			#next_states, rewards, done, _ = env.step((20,20))

			next_state_front, next_state_rear = next_states
			reward_front, reward_rear = rewards


			#update the epside reward
			episode_reward_front += reward_front
			episode_reward_rear  += reward_rear

			#remembering
			#front vehicle
			next_state_front = np.reshape(next_state_front, [1, state_size])
			agent_front.remember(state_front, action_front, reward_front, next_state_front, done)
			state_front = next_state_front

			#rear vehicle
			next_state_rear = np.reshape(next_state_rear, [1, state_size])
			agent_rear.remember(state_rear, action_rear, reward_rear, next_state_rear, done)
			state_rear = next_state_rear

			environment_state_memory[e % 250].append((env.variablesKinematicsFrontVehicle(), 
												env.variablesKinematicsRearVehicle(), 
												env.variablesEnvironment(), 
												((episode_reward_front, episode_reward_rear), (reward_front, reward_rear)),
												(action_front, action_delay_timer_front-1), (action_rear, action_delay_timer_rear-1)))


			if False:
				print('-----------')
				#vehicles: (pos, vel, acc, jer)
				#other vars: (headway, delta_headway)
				#print('current env: front vehicle :', ['{0:0.4f}'.format(j) for j in env.variablesKinematicsFrontVehicle()], '[', action_front, '(' + str(action_delay_timer_front-1) + '/' + str(action_delay_front) + ')', ']', '[',t + 1, '({0:0.3f} s)'.format((t+1) / 60), '/', e + 1,']')
				#print('             rear  vehicle :', ['{0:0.4f}'.format(j) for j in env.variablesKinematicsRearVehicle()],  '[', action_rear,  '(' + str(action_delay_timer_rear-1)  + '/' + str(action_delay_rear)  + ')', ']')
				#print('             other vars    :', [['{0:0.4f}'.format(j) for j in x] for x in env.variablesEnvironment()])
				#print('             reward        : [ (' + str(episode_reward_front) + ')  (' + str(reward_front) + ') ]  [ (' + str(episode_reward_rear) + ')  (' + str(reward_rear) + ') ]  ('+ str(extra['bound'])+')')

				print('current env: front vehicle :', ['{0:0.4f}'.format(j) for j in env.variablesKinematicsFrontVehicle()], '[', action_front, '(' + str(action_delay_timer_front-1) + '/' + str(action_delay_front) + ')', ']', '[',t + 1, '({0:0.3f} s)'.format((t+1) / 60), '/', e + 1,']', \
				    '\n             rear  vehicle :', ['{0:0.4f}'.format(j) for j in env.variablesKinematicsRearVehicle()],  '[', action_rear,  '(' + str(action_delay_timer_rear-1)  + '/' + str(action_delay_rear)  + ')', ']', \
				    '\n             other vars    :', [['{0:0.4f}'.format(j) for j in x] for x in env.variablesEnvironment()], \
				    '\n             reward        : [ (' + str(episode_reward_front) + ')  (' + str(reward_front) + ') ]  [ (' + str(episode_reward_rear) + ')  (' + str(reward_rear) + ') ]  ('+ str(extra['bound'])+')')
				#time.sleep(0.1)


			if done or t + 1 == STEP_LIMIT:
				agent_front.update_target_model()
				agent_rear.update_target_model()
				print('-----------')
				print('episode: {0:5d}/{1:5d} :: FRONT: score: {2}, e: {3:.2}'
					  .format(e+1, EPISODES, episode_reward_front, agent_front.epsilon))
				print('                     :: REAR:  score: {0}, e: {1:.2}'
					  .format(episode_reward_rear, agent_rear.epsilon))
				break
				

		#replaying memories
		if len(agent_front.memory) > BATCH_SIZE:
			agent_front.replay(BATCH_SIZE)
			
		if len(agent_rear.memory) > BATCH_SIZE:
			agent_rear.replay(BATCH_SIZE)


		#saving agents
		save_weights_to_file = True

		if save_weights_to_file and (e+1) % 250 == 0:
			print('saving weights to file')

			agent_front.save(PATH_TO_WEIGHTS+'front\\'+WEIGHT_NAME+'_front_'+str(e+1)+'.h5')
			agent_rear.save(PATH_TO_WEIGHTS+'rear\\'+WEIGHT_NAME+'_rear_'+str(e+1)+'.h5')

		#saving environment state memory (esm) to file
		save_esm_to_file = True

		if save_esm_to_file and (e+1) % 250 == 0:
			print('saving env state memory to file')

			with open(PROJECT_PATH+'env_state_mem\\adversary\\esm_'+VERSION+'_'+str(e+1)+'.txt', 'wb') as fp:
				pickle.dump(environment_state_memory, fp)
				environment_state_memory = []

	

	print('\n---------')
	print('DONE!!!')
	print('---------\n')


	

		



