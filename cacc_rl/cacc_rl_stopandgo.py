
from random import seed
from random import randint
import numpy as np
import time

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
from gym_cacc.envs import StopAndGo

#models
from models.dqn_model import DQNAgent



#hyper parameters
EPISODES	= 10005
STEP_LIMIT  = 2100 #steps
BATCH_SIZE	= 4048

#agent parameters
GAMMA			= 0.95
EPSILON			= 1.0
EPSILON_MIN		= 0.01
EPSILON_DECAY	= 0.99
LEARNING_RATE	= 0.00001
MEMORY_SIZE		= 2000

#vehicle parameters
#stats	:= (length, width, height, weigth, top_vel, top_acc, top_jerk)
#		:= (m, m, m, kg, m/s, m/s^2, m/s^3)
VEHICLE_STATISTICS = (4.67312, 1.81102, 1.43002, 1617.51, 55, 3, 1)
VEHICLE = Vehicle(statistics = VEHICLE_STATISTICS)

#environment parameters
TARGET_HEADWAY = 2
REACTION_TIME_LOWER_BOUND = 10 #frames (steps)
REACTION_TIME_UPPER_BOUND = 15 #frames (steps)

#names
VERSION = '1_1'
WEIGHT_NAME = 'cacc_rl_'+VERSION
PATH_TO_WEIGHTS = 'E:\\comp594\\Cooperative-Adaptive-CC-RL\\cacc_rl\\weights\\dqn_model_stopandgo\\'

print('starting main code')

if __name__ == '__main__':
	#create environment and agent
	env = gym.make('cacc-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	#setting environment variables and agent hyperparameters
	env.reset(VEHICLE, TARGET_HEADWAY)


	#weights_path = "E:\\comp594\\Cooperative-Adaptive-CC-RL\\cacc_rl\\weights\\dqn_model_stopandgo\\cacc_rl_1_1_250.h5"
	weights_path = None

	agent = DQNAgent(state_size, action_size, weights_path, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY, LEARNING_RATE, MEMORY_SIZE)


	#print('loading weights')
	#agent.load("E:\\comp594\\Cooperative-Adaptive-CC-RL\\cacc_rl\\weights\\dqn_model_stopandgo\\cacc_rl_1_1_25.h5")
	print('weights loaded')
	time.sleep(2)


	done = False

	environment_state_memory = []

	print('begin training')
	print('-------------------------------')

	for e in range(EPISODES):

		environment_state_memory.append([])


		#reset environment and get initial state
		state = env.reset()
		state = np.reshape(state, [1, state_size])

		
		episode_reward = 0
		
		action = 0

		action_delay = randint(10, 15)
		action_delay_timer = action_delay #allow agents to input on first step



		for time in range(STEP_LIMIT):


			#agent acts on environment's current state
			if action_delay_timer == action_delay:
				action_delay_timer = 0
				action_delay = randint(10, 15)
				action = agent.act(state)

			action_delay_timer  += 1


			#environment step
			next_state, reward, done, extra = env.step(action)

			#update the epside reward
			episode_reward += reward

			#remember
			next_state = np.reshape(next_state, [1, state_size])
			agent.remember(state, action, reward, next_state, done)
			state = next_state


			#print and save env state
			environment_state_memory[e].append((env.variablesKinematicsFrontVehicle(), 
												env.variablesKinematicsRearVehicle(), 
												env.variablesEnvironment(), 
												[episode_reward]))


			if True:
				print('-----------')
				#vehicles: (pos, vel, acc, jer)
				#other vars: (headway, delta_headway)
				print('current env: front vehicle :', [float(i) for i in ['{0:0.4f}'.format(j) for j in env.variablesKinematicsFrontVehicle()]], '[',time + 1, '({0:0.3f} s)'.format((time+1) / 60), '/', e + 1,']')
				print('             rear  vehicle :', [float(i) for i in ['{0:0.4f}'.format(j) for j in env.variablesKinematicsRearVehicle()]],  '[', action, ']')
				print('             other vars    :', [[float(i) for i in ['{0:0.4f}'.format(j) for j in x]] for x in env.variablesEnvironment()])
				print('             other vars2   :', [float(i) for i in ['{0:0.4f}'.format(j) for j in env.variablesOthers()]])
				print('             reward        : (' + str(episode_reward) + ')  (' + str(reward) + ')  ('+ str(extra['bound'])+')')


			#simulation complete
			if done or time + 1 == STEP_LIMIT:
				agent.update_target_model()
				print('-----------')
				print('episode: {0:3d}/{1:3d} :: REAR: score: {2}, e: {3:.2}'
					  .format(e+1, EPISODES, episode_reward, agent.epsilon))
				
				break

			if len(agent.memory) > BATCH_SIZE:
				agent.replay(BATCH_SIZE)

		
		#saving agents
		if e % 50 == 0:
			agent.save(PATH_TO_WEIGHTS+WEIGHT_NAME+'_'+str(e)+'.h5')


		#saving env state and variables
		save_to_file = True

		if save_to_file and e % 250 == 0:
			print('saving env state memory to file')

			with open('env_state_mem\\stopandgo\\env_state_mem_'+str(e)+'.txt', 'wb') as fp:
				pickle.dump(environment_state_memory, fp)


	print('\n---------')
	print('DONE!!!')
	print('---------\n')

	

