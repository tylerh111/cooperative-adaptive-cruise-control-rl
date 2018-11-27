
import os
import sys
import random
import numpy as np
import time
import pickle

from tqdm import tqdm

#params
EPOCHS = 100
BATCH_SIZE	= 32
BATCHES_PER_EPOCH = 256
VALIDATION_SPLIT = 0.20

NUM_PARTITIONS = 200
NUM_CLASSES = 21
NUM_RUNS = 3

#hyper params
LEARNING_RATE          = 1e-2
LEARNING_RATE_DECAY    = 1e-6
LEARNING_RATE_MOMENTUM = 0.9

#dimensions
TIME_SIZE = 60
INPUT_SHAPE  = (TIME_SIZE, 7)
OUTPUT_SHAPE = (NUM_CLASSES)


#names
#project path
PROJECT_PATH = 'E:\\comp594\\Cooperative-Adaptive-CC-RL\\cacc_rl\\'
DIR_SEP = '\\'

VEHICLE_TRAINED = 'rear'
MODEL_TYPE = 'rnn'

DATA_PATH_ORIGIN = 'F:\\workspace\\cacc_rl\\adversary_old_reward\\'
DATA_PATH = DATA_PATH_ORIGIN + MODEL_TYPE + DIR_SEP

DATA_BATCHES_PATH = DATA_PATH + 'data_batches' + DIR_SEP + VEHICLE_TRAINED + DIR_SEP
DATA_BATCH_NAME = 'batch_dnn_adg_v1_0.txt'
DATA_BATCH_PATH = DATA_BATCHES_PATH + DATA_BATCH_NAME

DATA_SETS_PATH = DATA_PATH + 'data_sets' + DIR_SEP + VEHICLE_TRAINED + DIR_SEP
DATA_SET_NAME = DATA_BATCH_NAME[:-4]   # it's a directory
DATA_SET_PATH = DATA_SETS_PATH + DATA_SET_NAME + DIR_SEP
DATA_PARTITION_NAME = 'partition_' # number follows
DATA_PARTITION_PATH = DATA_SET_PATH + DATA_PARTITION_NAME

VERSION_MAJOR_NUMBER = 1
VERSION_MINOR_NUMBER = 1
VERSION_NAME = MODEL_TYPE + '_' + VEHICLE_TRAINED + '_at_v' + str(VERSION_MAJOR_NUMBER) + '_' + str(VERSION_MINOR_NUMBER)

WEIGHTS_PATH = PROJECT_PATH + 'weights\\' + MODEL_TYPE + '_model_adversary\\' + VEHICLE_TRAINED + DIR_SEP
WEIGHT_NAME = VERSION_NAME + '_weights_{epoch:04d}.hdf5'
#WEIGHT_PATH = WEIGHTS_PATH + WEIGHT_NAME

TENSORBOARD_LOG_DIR_PATH = DATA_PATH_ORIGIN + 'tb_logs' + DIR_SEP

LOAD_FROM_DATA_BATCH = True

ALREADY_LOADED_DATA_BATCH = os.path.exists(DATA_SET_PATH)






#-----------------
#BUILDING DATASET
#-----------------

if LOAD_FROM_DATA_BATCH and not ALREADY_LOADED_DATA_BATCH:

	print('loading data batch at [' + DATA_BATCH_PATH + ']...')#, end='')
	with open(DATA_BATCH_PATH, 'rb') as f:
		data_batch = pickle.load(f)

	print('done.')
	

	#print('creating data set...')#, end='')
	data_set = [[] for i in range(NUM_CLASSES)]

	get_action = lambda frame: frame[5]
	get_reward = lambda frame: frame[6]

	for sim in tqdm(data_batch, desc='Creating data set'):
		if len(sim) > 480:
			j = TIME_SIZE

			for i in range(0, len(sim) - TIME_SIZE - 1):
				if get_reward(sim[j+1]) > 0:
					x = sim[i:j]
					y = get_action(sim[j+1])

					data_set[y].append(x)

				j += 1

	#print('done')


	#print('partitioning data set...')#, end='')

	data_partitions = [([], []) for i in range(NUM_PARTITIONS)]

	for i in tqdm(range(NUM_CLASSES), desc='Partitioning data set'):
		for p in data_set[i]:
			r = random.randrange(0, NUM_PARTITIONS)
			data_partitions[r][0].append(p)
			data_partitions[r][1].append(i)

	#print('done.')


	#print('saving data sets...')#, end='')
	os.makedirs(DATA_SET_PATH)

	i = -1
	for partition in tqdm(data_partitions, desc='Saving data sets'):
		i+=1
		with open(DATA_PARTITION_PATH + str(i) + '.txt', 'wb') as f:
			pickle.dump(partition, f)

	#print('done.')


	del data_partitions
	del partition
	del sim
	del p
	del data_batch
	del data_set

elif ALREADY_LOADED_DATA_BATCH:
	print('data set exists for current data batch')

#else:
#	print('retreiving data set paths...')#, end='')
#	list_paths = []
#	for root, dirs, files in os.walk(DATA_SETS_PATH):
#		for file in files:
#			filepath = os.path.join(root, file)
#			list_paths.append(filepath)

#	print('done.')


#	print('partitioning data set...')#, end='')

#	data_partitions = [([], []) for i in range(NUM_PARTITIONS)]

#	for filepath in list_paths:
#		with open(filepath, 'rb') as f:
#			data_batch = pickle.load(f)

#			for i in range(NUM_CLASSES):
#				for p in data_batch[i]:
#					r = random.randrange(0, NUM_PARTITIONS)
#					data_partitions[r][0].append(p)
#					data_partitions[r][1].append(i)

#	print('done.')










#-----------------
#BUILDING MODEL
#-----------------

import tensorflow as tf

#import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import LSTM, CuDNNLSTM, Dense, Input, Reshape, Flatten, Dropout, BatchNormalization, GlobalMaxPool2D, Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers, losses, activations, models, metrics, utils
from tensorflow.keras import backend as K


def build_model(weights_path = None):

	model = Sequential()

	model.add(LSTM(128, input_shape=INPUT_SHAPE, activation=K.sigmoid, return_sequences=True, name='lstm_fc1'))
	model.add(Dropout(0.2, name='dropout_fc1'))
	model.add(LSTM(128, activation=K.sigmoid, name='lstm_fc2'))
	model.add(Dropout(0.2, name='dropout_fc2'))
	model.add(Dense(32, activation=K.sigmoid, name='dense_fc3'))
	model.add(Dropout(0.2, name='dropout_fc3'))
	model.add(Dense(NUM_CLASSES, activation=K.softmax, name='actions'))
	
	if weights_path is not None:
		model.load_weights(weights_path)

	opt = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE_DECAY)
	#opt = SGD(lr=LEARNING_RATE, decay=LEARNING_RATE_DECAY, momentum=LEARNING_RATE_MOMENTUM)

	model.compile(loss='categorical_crossentropy',
					optimizer=opt,
					metrics=['accuracy'])

	return model


print('building model...')#, end='')
model = build_model()
print('done')

print('Model Summary')
model.summary()




#from keras_tqdm import TQDMCallback



#-----------------
#TRAINING MODEL
#-----------------

def train_model(r, train_partition, valid_partition):
	
	#PREPARE A DATASET FOR USE
	print('preparing training set...')

	train_X, train_y = train_partition
	train_X = np.array(train_X)
	train_y = np.array(train_y)
	train_y = utils.to_categorical(train_y, num_classes=NUM_CLASSES)

	print('done.')


	print('preparing validation set...')

	valid_X, valid_y = valid_partition
	sampled_valid_X = []
	sampled_valid_y = []

	sample_ndxs = random.sample(range(0, len(valid_X)), int(len(valid_X) * VALIDATION_SPLIT))
	for i in sample_ndxs:
		sampled_valid_X.append(valid_X[i])
		sampled_valid_y.append(valid_y[i])

	sampled_valid_X = np.array(sampled_valid_X)
	sampled_valid_y = np.array(sampled_valid_y)
	sampled_valid_y = utils.to_categorical(sampled_valid_y, num_classes=NUM_CLASSES)

	print('done.')


	#TRAIN
	print('begin training')
	print('--------------------')

	callback_list = [
		ModelCheckpoint(new_weight_path, monitor='val_acc', verbose=1, period=20), 
		TensorBoard(log_dir=new_tensorboard_log_dir_path, histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True),
		#TQDMCallback(leave_inner=False, leave_outer=True)
	]

	#train model on dataset
	history = model.fit(train_X, train_y,
						batch_size=BATCH_SIZE, 
						epochs=(r+1)*EPOCHS, 
						verbose=1,
						callbacks=callback_list,
						#validation_split=VALIDATION_SPLIT,)
						validation_data=(sampled_valid_X, sampled_valid_y),
						initial_epoch=r*EPOCHS)

	#print(history)








for r in range(0, NUM_RUNS):
	print('=' * 142)

	#paths to logs and weights
	new_weight_dir_path = WEIGHTS_PATH + VERSION_NAME + '_run_' + str(r) + DIR_SEP
	new_weight_path     = new_weight_dir_path + WEIGHT_NAME
	new_tensorboard_log_dir_path = TENSORBOARD_LOG_DIR_PATH + VERSION_NAME + '_run_' + str(r) + DIR_SEP
		
	if not os.path.exists(new_weight_dir_path):
		os.makedirs(new_weight_dir_path)

	if not os.path.exists(new_tensorboard_log_dir_path):
		os.makedirs(new_tensorboard_log_dir_path)
	
	
	
	#PICK DATA PARTITION
	partition_ndx = random.sample(range(0, NUM_PARTITIONS), 2)
	train_partition_ndx = partition_ndx[0]
	valid_partition_ndx = partition_ndx[1]

	with open(DATA_PARTITION_PATH + str(train_partition_ndx) + '.txt', 'rb') as f:
		train_partition = pickle.load(f)

	with open(DATA_PARTITION_PATH + str(valid_partition_ndx) + '.txt', 'rb') as f:
		valid_partition = pickle.load(f)


	def normalize_data_set(x):
		for period in tqdm(x[0], desc='Normalizing data'):
			for frame in period:
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


	normalize_data_set(train_partition)
	normalize_data_set(valid_partition)

	train_model(r, train_partition, valid_partition)


