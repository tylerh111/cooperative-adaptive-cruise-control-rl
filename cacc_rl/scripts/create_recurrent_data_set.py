

import os
import sys
import random
import time
import pickle


#names and paths
PROJECT_PATH = 'E:\\comp594\\Cooperative-Adaptive-CC-RL\\cacc_rl\\'
DIR_SEP = '\\'

VEHICLE_TRAINED = 'rear'

DATA_PATH = 'F:\\workspace\\cacc_rl\\adversary_old_reward\\dnn\\'

DATA_BATCHES_PATH = DATA_PATH + 'data_batches' + DIR_SEP + VEHICLE_TRAINED + DIR_SEP
DATA_BATCH_PATH = DATA_BATCHES_PATH + 'batch_adg_v1_0_rnn.txt'

DATA_SETS_PATH = DATA_PATH + 'data_sets' + DIR_SEP + VEHICLE_TRAINED + DIR_SEP
DATA_SET_VERSION = 1
DATA_SET_NAME = 'ds_v' + str(DATA_SET_VERSION) + '.txt'
DATA_SET_PATH = DATA_SETS_PATH + DATA_SET_NAME

NUM_CLASSES = 21

VEHICLE = 'rear'


print('loading data batch at [' + DATA_BATCH_PATH + ']...')
with open(DATA_BATCH_PATH, 'rb') as f:
	data_batch = pickle.load(f)

print('done')


print('creating data set...')#, end='')



data_set = [[] for i in range(NUM_CLASSES)]

for sim in data_batch:










with open(DATA_SET_PATH + DATA_SET_NAME, 'wb') as f:
	pickle.dump(data_set, f)

print('done.')

#for filepath in list_paths:
#	label = get_class_from_path(filepath)
	
#	with open(filepath, 'rb') as f:
#		data_batch = pickle.load(f)

#		for data_point in data_batch:
#			data_set[label].append(data_point)







