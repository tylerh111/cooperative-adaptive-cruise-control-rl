
from random import seed
from random import randint
import numpy as np
import time
import threading
import h5py

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
from models.rnn_model import build_model


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

#DATA_PATH = 'F:\\workspace\\cacc_rl\\adversary_old_reward\\'
#DATA_BATCHES_DNN_PATH_FRONT = DATA_PATH + 'dnn' + DIR_SEP + 'data_batches' + DIR_SEP + 'front' + DIR_SEP
#DATA_BATCHES_DNN_PATH_REAR  = DATA_PATH + 'dnn' + DIR_SEP + 'data_batches' + DIR_SEP + 'rear'  + DIR_SEP
#DATA_BATCHES_RNN_PATH_FRONT = DATA_PATH + 'rnn' + DIR_SEP + 'data_batches' + DIR_SEP + 'front' + DIR_SEP
#DATA_BATCHES_RNN_PATH_REAR  = DATA_PATH + 'rnn' + DIR_SEP + 'data_batches' + DIR_SEP + 'rear'  + DIR_SEP

#DATA_BATCH_NAME_FRONT = 'batch_' + VERSION_NAME + '.txt'
#DATA_BATCH_NAME_REAR  = 'batch_' + VERSION_NAME + '.txt'

#DATA_BATCH_DNN_PATH_FRONT = DATA_BATCHES_DNN_PATH_FRONT + DATA_BATCH_NAME_FRONT
#DATA_BATCH_DNN_PATH_REAR  = DATA_BATCHES_DNN_PATH_REAR  + DATA_BATCH_NAME_REAR
#DATA_BATCH_RNN_PATH_FRONT = DATA_BATCHES_RNN_PATH_FRONT + DATA_BATCH_NAME_FRONT
#DATA_BATCH_RNN_PATH_REAR  = DATA_BATCHES_RNN_PATH_REAR  + DATA_BATCH_NAME_REAR


#hyper parameters
EPISODES	= 1000
STEP_LIMIT  = 2000 #steps
#BATCH_SIZE	= 32

#vehicle parameters
#stats	:= (length, width, height, weigth, top_vel, top_acc, top_jerk)
#		:= (m, m, m, kg, m/s, m/s^2, m/s^3)
VEHICLE_STATISTICS = (4, 2, 1.5, 1617.51, 55, 3, 12)
VEHICLE = Vehicle(statistics = VEHICLE_STATISTICS)

#environment parameters
TARGET_HEADWAY = 2
GRANULARITY = 21
NUM_CLASSES = GRANULARITY

INIT_VELOCITY = 25
INIT_ACCELERATION = 0
ALPHA_VELOCITY = 4
ALPHA_ACCELERATION = 2

REACTION_TIME_LOWER_BOUND = 10 #frames (steps)
REACTION_TIME_UPPER_BOUND = 15 #frames (steps)

#agent 
#hyper params
LEARNING_RATE          = 1e-2
LEARNING_RATE_DECAY    = 1e-6
LEARNING_RATE_MOMENTUM = 0.9

#dimensions
TIME_SIZE = 60
INPUT_SHAPE  = (TIME_SIZE, 7)
OUTPUT_SHAPE = (NUM_CLASSES)




def getListFromLast60AndNormalize(lst, ndx):
	res = []
	for i in range(0,60):
		res.append(lst[ndx % 60])
		ndx += 1

		
	for frame in res:
		if frame[0] <= 0   or frame[0] >= 6:  frame[0] = 0   if frame[0] <= 0   else 6
		frame[0] /= 6

		if frame[1] <= -12 or frame[1] >= 12: frame[1] = -12 if frame[1] <= -12 else 12
		frame[1] /= 12

		if frame[2] <= -12 or frame[2] >= 12: frame[2] = -12 if frame[2] <= -12 else 12
		frame[2] /= 12

		if frame[3] <= 0   or frame[3] >= 6:  frame[3] = 0   if frame[3] <= 0   else 55
		frame[3] /= 55

		if frame[4] <= -12 or frame[4] >= 12: frame[4] = -12 if frame[4] <= -12 else 12
		frame[4] /= 12


	return res




#create environment and agent
env = gym.make('cacc-v0')

#setting environment variables and agent hyperparameters
env.reset(VEHICLE, TARGET_HEADWAY, GRANULARITY, INIT_VELOCITY, ALPHA_VELOCITY, INIT_ACCELERATION, ALPHA_ACCELERATION)
	
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


weights_path = PROJECT_PATH + 'weights\\rnn_model_adversary\\rear\\rnn_rear_at_v1_1_run_0\\rnn_rear_at_v1_1_weights_0020.hdf5'
model = build_model(INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE, LEARNING_RATE_DECAY, weights_path)


#with h5py.File(weights_path, mode='r') as f:
#	# new file format
#	layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]


done = False

esm = []


print('begin data generation')
print('-------------------------------')

#amm = [] #action mode memory

for e in range(EPISODES):

	esm.append([])

	#reset environment and get initial state
	state = env.reset(VEHICLE, TARGET_HEADWAY, GRANULARITY, INIT_VELOCITY, ALPHA_VELOCITY, INIT_ACCELERATION, ALPHA_ACCELERATION)
	state = np.reshape(state,  [1, state_size])

	action = 10
	episode_reward = 0

	action_delay = randint(REACTION_TIME_LOWER_BOUND, REACTION_TIME_UPPER_BOUND)
	action_delay_timer = action_delay #allow agents to input on first step

	action_mode = [0]*21

	past_60_steps = [[] for i in range(0,60)]

	for t in range(STEP_LIMIT):
			
		#agents act on environment's current state
		if t >= 60 and action_delay_timer == action_delay:
			action_delay_timer = 0
			action_delay = randint(REACTION_TIME_LOWER_BOUND, REACTION_TIME_UPPER_BOUND)
			#action = randint(0, GRANULARITY - 1)
			# t % 60 is the index into past_60_steps that is the beginning of the past 60 steps
			input = getListFromLast60AndNormalize(past_60_steps, t % 60)
			input = np.array([input])
			action_prediction = model.predict_on_batch(input)[0]
			action = np.argmax(action_prediction)
			action_mode[action] += 1
		
		elif action_delay_timer != action_delay:
			action_delay_timer += 1


		#environment step
		next_state, reward, done, extra = env.step(action)
		
		#update the epside reward
		episode_reward += reward


		past_60_steps[t % 60] = state.tolist()[0] + [action, reward]

		next_state = np.reshape(next_state, [1, state_size])
		
		
		state = next_state


		esm[e].append((env.variablesKinematicsFrontVehicle(), 
						env.variablesKinematicsRearVehicle(), 
						env.variablesEnvironment(), 
						env.variablesOthers(),
						[episode_reward, reward],
						[action, action_delay_timer-1]))


		if t % 10 == 0: #True:
			print('-----------')
			print('current env: front vehicle :', ['{0:0.4f}'.format(j) for j in env.variablesKinematicsFrontVehicle()], '[',t + 1, '({0:0.3f} s)'.format((t+1) / 60), '/', e + 1,']', \
				'\n             rear  vehicle :', ['{0:0.4f}'.format(j) for j in env.variablesKinematicsRearVehicle()],  '[', action,  '(' + str(action_delay_timer-1)  + '/' + str(action_delay)  + ')', ']', \
				'\n             other vars    :', [['{0:0.4f}'.format(j) for j in x] for x in env.variablesEnvironment()], \
				'\n             reward        : [ (' + str(episode_reward) + ')  (' + str(reward) + ') ]  ('+ str(extra['bound'])+')')
			time.sleep(0.1)


		if (done or t + 1 == STEP_LIMIT) and e % 1 == 0:
			print('-----------------------------------------------------------------------------------------------------')
			print('episode: {0:5d}/{1:5d} :: STATS     :'.format(e+1, EPISODES), [['{0:0.4f}'.format(j) for j in x] for x in env.variablesEnvironment()[1:]], '('+str(t)+')')
			print('                      :: SEEDS     :', env.getSeed())
			print('                      :: REWARDS   :', (episode_reward))
			print('                      :: REAR  VALS:', action_mode)

			time.sleep(0.1)
			break
			
		continue #end of time step

	#saving env state and variables
	save_to_file = True

	if save_to_file and e % 250 == 0:
		print('saving env state memory to file')

		with open('env_state_mem\\stopandgo\\env_state_mem_'+str(e)+'.txt', 'wb') as fp:
			pickle.dump(esm, fp)

	continue #end of episode






print('\n---------')
print('DONE!!!')
print('---------\n')







