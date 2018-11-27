
from random import seed
from random import randint
import numpy as np
import time
import threading

import pickle


#keras (not needed)
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
#from keras import backend as K

#import tensorflow as tf

#gym
import gym
from gym_cacc import Vehicle
from gym_cacc.envs import Adversary

#models
#from models.dqn_model import DQNAgent


#project path
PROJECT_PATH = 'E:\\comp594\\Cooperative-Adaptive-CC-RL\\cacc_rl\\'

DIR_SEP = '\\'

#names
VERSION_MODE_LIST = ['rnn', 'dnn']
VERSION_MODE = 'rnn'
VERSION_MAJOR_NUMBER = 1
VERSION_MINOR_NUMBER = 1
VERSION_NAME = VERSION_MODE + '_adg_v'+str(VERSION_MAJOR_NUMBER) + '_' +str(VERSION_MINOR_NUMBER) 

IS_VERSION_MODE_DNN = VERSION_MODE == 'dnn'
IS_VERSION_MODE_RNN = VERSION_MODE == 'rnn'

DATA_PATH = 'F:\\workspace\\cacc_rl\\adversary_old_reward\\'
DATA_BATCHES_DNN_PATH_FRONT = DATA_PATH + 'dnn' + DIR_SEP + 'data_batches' + DIR_SEP + 'front' + DIR_SEP
DATA_BATCHES_DNN_PATH_REAR  = DATA_PATH + 'dnn' + DIR_SEP + 'data_batches' + DIR_SEP + 'rear'  + DIR_SEP
DATA_BATCHES_RNN_PATH_FRONT = DATA_PATH + 'rnn' + DIR_SEP + 'data_batches' + DIR_SEP + 'front' + DIR_SEP
DATA_BATCHES_RNN_PATH_REAR  = DATA_PATH + 'rnn' + DIR_SEP + 'data_batches' + DIR_SEP + 'rear'  + DIR_SEP

DATA_BATCH_NAME_FRONT = 'batch_' + VERSION_NAME + '.txt'
DATA_BATCH_NAME_REAR  = 'batch_' + VERSION_NAME + '.txt'

DATA_BATCH_DNN_PATH_FRONT = DATA_BATCHES_DNN_PATH_FRONT + DATA_BATCH_NAME_FRONT
DATA_BATCH_DNN_PATH_REAR  = DATA_BATCHES_DNN_PATH_REAR  + DATA_BATCH_NAME_REAR
DATA_BATCH_RNN_PATH_FRONT = DATA_BATCHES_RNN_PATH_FRONT + DATA_BATCH_NAME_FRONT
DATA_BATCH_RNN_PATH_REAR  = DATA_BATCHES_RNN_PATH_REAR  + DATA_BATCH_NAME_REAR



#hyper parameters
EPISODES	= 10000
STEP_LIMIT  = 3000 #steps
#BATCH_SIZE	= 32

#vehicle parameters
#stats	:= (length, width, height, weigth, top_vel, top_acc, top_jerk)
#		:= (m, m, m, kg, m/s, m/s^2, m/s^3)
VEHICLE_STATISTICS = (4, 2, 1.5, 1617.51, 55, 3, 12)
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






#create environment and agent
env = gym.make('cacc-adversary-v0')

#setting environment variables and agent hyperparameters
env.reset(VEHICLE, TARGET_HEADWAY, GRANULARITY, INIT_VELOCITY, ALPHA_VELOCITY, INIT_ACCELERATION, ALPHA_ACCELERATION)
	
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


done = False

#esm = []

if IS_VERSION_MODE_DNN:
	action_classes_front = [list() for i in range(GRANULARITY)]
	action_classes_rear  = [list() for i in range(GRANULARITY)]

if IS_VERSION_MODE_RNN:
	simulation_memory_front = []
	simulation_memory_rear  = []


print('begin data generation')
print('-------------------------------')

#amm = [] #action mode memory

for e in range(EPISODES):

	#esm.append([])

	#reset environment and get initial state
	state = env.reset(VEHICLE, TARGET_HEADWAY, GRANULARITY, INIT_VELOCITY, ALPHA_VELOCITY, INIT_ACCELERATION, ALPHA_ACCELERATION)
	state_front, state_rear = state

	state_front = np.reshape(state_front, [1, state_size])
	state_rear  = np.reshape(state_rear,  [1, state_size])

	action_front = 0
	action_rear  = 0
	action = (action_front, action_rear)

	episode_reward_front = 0
	episode_reward_rear  = 0

	action_delay_front = randint(REACTION_TIME_LOWER_BOUND, REACTION_TIME_UPPER_BOUND)
	action_delay_rear  = randint(REACTION_TIME_LOWER_BOUND, REACTION_TIME_UPPER_BOUND)

	#allow agents to input on first step
	action_delay_timer_front = action_delay_front 
	action_delay_timer_rear  = action_delay_rear

	action_mode_front = [0]*21
	action_mode_rear  = [0]*21
	
	
	if IS_VERSION_MODE_RNN:
		simulation_memory_front.append([])
		simulation_memory_rear.append([])
	

	for t in range(STEP_LIMIT):
			
		#agents act on environment's current state
		#front agent
		if action_delay_timer_front == action_delay_front:
			action_delay_timer_front = 0
			action_delay_front = randint(REACTION_TIME_LOWER_BOUND, REACTION_TIME_UPPER_BOUND)
			action_front = randint(0, GRANULARITY - 1)
			action_mode_front[action_front] += 1
			
		#rear agent
		if action_delay_timer_rear == action_delay_rear:
			action_delay_timer_rear = 0
			action_delay_rear = randint(REACTION_TIME_LOWER_BOUND, REACTION_TIME_UPPER_BOUND)
			action_rear = randint(0, GRANULARITY - 1)
			action_mode_rear[action_rear] += 1

		action = (action_front, action_rear)

		action_delay_timer_front += 1
		action_delay_timer_rear  += 1

		#environment step
		next_states, rewards, done, extra = env.step(action)
			
		next_state_front, next_state_rear = next_states
		reward_front, reward_rear = rewards

		#update the epside reward
		episode_reward_front += reward_front
		episode_reward_rear  += reward_rear


		#SAVE TO MEMORY
		if IS_VERSION_MODE_DNN:
			if reward_front > 0:
				action_classes_front[action_front].append(next_state_front.tolist())
			elif reward_rear > 0:
				action_classes_rear[action_rear].append(next_state_rear.tolist())
		
		if IS_VERSION_MODE_RNN:
			simulation_memory_front[e].append(state_front.tolist()[0] + [action_front, reward_front])
			simulation_memory_rear[e].append( state_rear.tolist()[0]  + [action_rear,  reward_rear ])


		next_state_front = np.reshape(next_state_front, [1, state_size])
		next_state_rear  = np.reshape(next_state_rear,  [1, state_size])
		
		
		state_front = next_state_front
		state_rear  = next_state_rear


		#esm[e].append((env.variablesKinematicsFrontVehicle(), 
		#				env.variablesKinematicsRearVehicle(), 
		#				env.variablesEnvironment(), 
		#				((episode_reward_front, episode_reward_rear), (reward_front, reward_rear)),
		#				(action_front, action_delay_timer_front-1), (action_rear, action_delay_timer_rear-1)))


		if False:
			print('-----------')
			print('current env: front vehicle :', ['{0:0.4f}'.format(j) for j in env.variablesKinematicsFrontVehicle()], '[', action_front, '(' + str(action_delay_timer_front-1) + '/' + str(action_delay_front) + ')', ']', '[',t + 1, '({0:0.3f} s)'.format((t+1) / 60), '/', e + 1,']', \
				'\n             rear  vehicle :', ['{0:0.4f}'.format(j) for j in env.variablesKinematicsRearVehicle()],  '[', action_rear,  '(' + str(action_delay_timer_rear-1)  + '/' + str(action_delay_rear)  + ')', ']', \
				'\n             other vars    :', [['{0:0.4f}'.format(j) for j in x] for x in env.variablesEnvironment()], \
				'\n             reward        : [ (' + str(episode_reward_front) + ')  (' + str(reward_front) + ') ]  [ (' + str(episode_reward_rear) + ')  (' + str(reward_rear) + ') ]  ('+ str(extra['bound'])+')')
			#time.sleep(0.1)


		if (done or t + 1 == STEP_LIMIT) and e % 1 == 0:
			print('-----------------------------------------------------------------------------------------------------')
			print('episode: {0:5d}/{1:5d} :: STATS     :'.format(e+1, EPISODES), [['{0:0.4f}'.format(j) for j in x] for x in env.variablesEnvironment()[1:]], '('+str(t)+')')
			print('                      :: SEEDS     :', env.getSeed())
			print('                      :: REWARDS   :', (episode_reward_front, episode_reward_rear))
			print('                      :: FRONT VALS:', action_mode_front)
			print('                      :: REAR  VALS:', action_mode_rear)

			break
			
		continue #end of time step

	continue #end of episode




#SAVE TO FILE
print('saving to file...')

if IS_VERSION_MODE_DNN:
	with open(DATA_BATCH_DNN_PATH_FRONT, 'wb') as f:
		pickle.dump(action_classes_front, f)

	with open(DATA_BATCH_DNN_PATH_REAR,  'wb') as f:
		pickle.dump(action_classes_rear,  f)


if IS_VERSION_MODE_RNN:
	with open(DATA_BATCH_RNN_PATH_FRONT, 'wb') as f:
		pickle.dump(simulation_memory_front, f)

	with open(DATA_BATCH_RNN_PATH_REAR,  'wb') as f:
		pickle.dump(simulation_memory_rear,  f)






print('\n---------')
print('DONE!!!')
print('---------\n')







