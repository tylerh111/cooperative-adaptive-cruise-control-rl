
import os
import sys
import random
import numpy as np
import time
import pickle

#params
EPOCHS = 300
BATCH_SIZE	= 32
BATCHES_PER_EPOCH = 256
VALIDATION_SPLIT = 0.20

NUM_PARTITIONS = 50
NUM_CLASSES = 21
NUM_RUNS = 10

#hyper params
LEARNING_RATE       = 1e-2
LEARNING_RATE_DECAY = 1e-4


#dimensions
INPUT_SHAPE  = (5)
OUTPUT_SHAPE = (NUM_CLASSES)



#names
#project path
PROJECT_PATH = 'E:\\comp594\\Cooperative-Adaptive-CC-RL\\cacc_rl\\'
DIR_SEP = '\\'

VEHICLE_TRAINED = 'rear'
MODEL_TYPE = 'dnn'

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
VERSION_MINOR_NUMBER = 0
VERSION_NAME = MODEL_TYPE + '_' + VEHICLE_TRAINED + '_at_v' + str(VERSION_MAJOR_NUMBER) + '_' + str(VERSION_MINOR_NUMBER)

WEIGHTS_PATH = PROJECT_PATH + 'weights\\' + MODEL_TYPE + '_model_adversary\\' + VEHICLE_TRAINED + DIR_SEP
WEIGHT_NAME = VERSION_NAME + '_weights.{epoch:04d}.hdf5'
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


	print('partitioning data set...')#, end='')

	data_partitions = [([], []) for i in range(NUM_PARTITIONS)]

	for i in range(NUM_CLASSES):
		for p in data_batch[i]:
			r = random.randrange(0, NUM_PARTITIONS)
			data_partitions[r][0].append(p)
			data_partitions[r][1].append(i)

	print('done.')

	print('saving data sets...')#, end='')
	os.makedirs(DATA_SET_PATH)

	i = -1
	for partition in data_partitions:
		i+=1
		with open(DATA_PARTITION_PATH + str(i) + '.txt', 'wb') as f:
			pickle.dump(partition, f)

	print('done.')

	del data_partitions
	del partition
	del data_batch
	del p

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

import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Convolution2D, Dense, Input, Reshape, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalMaxPool2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers, losses, activations, models, metrics
from tensorflow.keras import backend as K



def build_model(weights_path = None):

	model = Sequential()

	model.add(Dense(32, input_dim=INPUT_SHAPE, activation=K.sigmoid, name='dense_fc1'))
	model.add(Dropout(0.3, name='dropout_fc1'))
	model.add(Dense(64, activation=K.sigmoid, name='dense_fc2'))
	model.add(Dropout(0.3, name='dropout_fc2'))
	model.add(Dense(32, activation=K.sigmoid, name='dense_fc3'))
	model.add(Dropout(0.3, name='dropout_fc3'))
	model.add(Dense(NUM_CLASSES, activation=K.softmax, name='actions'))

	if weights_path is not None:
		model.load_weights(weights_path)

	opt = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE_DECAY)

	model.compile(loss='categorical_crossentropy',
					optimizer=opt,
					metrics=['accuracy'])

	return model


print('building model...')#, end='')
model = build_model()
print('done')

print('Model Summary')
model.summary()







#-----------------
#TRAINING MODEL
#-----------------

def train_model(r, train_partition, valid_partition):
	
	#PREPARE A DATASET FOR USE
	print('preparing training set...')

	train_X, train_y = train_partition
	train_X = np.array(train_X)
	train_y = np.array(train_y)
	train_y = keras.utils.to_categorical(train_y, num_classes=NUM_CLASSES)

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
	sampled_valid_y = keras.utils.to_categorical(sampled_valid_y, num_classes=NUM_CLASSES)

	print('done.')


	#TRAIN
	print('begin training')
	print('--------------------')

	callback_list = [
		ModelCheckpoint(new_weight_path, monitor='val_acc', verbose=1, period=10), 
		TensorBoard(log_dir=new_tensorboard_log_dir_path, histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True)
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
		for frame in x[0]:
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






##getting vehicle data paths
#print('retreiving data paths...')#, end='')
#list_paths = []
#for root, dirs, files in os.walk(DATA_PATH + VEHICLE_TRAINED + DIR_SEP):
#	for file in files:
#		filepath = os.path.join(root, file)
#		list_paths.append(filepath)

#print('done.')


#def get_class_from_path(filepath):
#	return int(os.path.dirname(filepath).split(os.sep)[-1])


#print('creating data sets...')#, end='')
#train_set = [[] for i in range(NUM_CLASSES)]
##train_set_ex1 = [[] for i in range(NUM_CLASSES)]
##train_set_ex2 = [[] for i in range(NUM_CLASSES)]
##train_set_ex3 = [[] for i in range(NUM_CLASSES)]

#valid_set = [[] for i in range(NUM_CLASSES)]

#for filepath in list_paths:
#	label = get_class_from_path(filepath)
#	if random.randint(0,4) != 0:
#		train_set[label].append(filepath)

#		#r = random.randint(0,2)
#		#if r == 0:
#		#	train_set_ex1[label].append(filepath)
#		#elif r == 1:
#		#	train_set_ex2[label].append(filepath)
#		#elif r == 2:
#		#	train_set_ex3[label].append(filepath)

#	else:
#		valid_set[label].append(filepath)	


#partition = {'train': [train_set],
#			 'valid': [valid_set],}

#print('done.')



##def load_and_sample_data(path, sample_size):
##	with open(path, 'rb') as fp:
##		data = pickle.load(fp)
##		sampled_data = random.sample(data, sample_size)
##		return True, sampled_data

##	return False, None

#def load_data_point(path):
#	with open(path, 'rb') as fp:
#		data = pickle.load(fp)
#		data_point = random.choice(data)
#		return True, data_point

#	return False, None


#class DataGenerator:

#	def __init__(self, dim_num = 1000, dim_state = 5, batch_size = 40, batches_per_epoch = 100, nclass=21):
#		self.dim_num = dim_num
#		self.dim_state = dim_state
#		self.batch_size = batch_size
#		self.batches_per_epoch = batches_per_epoch
#		self.nclass = nclass

#	def generate(self, list_ids):
#		# Generates batches of samples
#		# Infinite loop
#		while 1:
#			# Generate batches
#			imax = self.batches_per_epoch
#			for i in range(imax):
#				# Generate data
#				X, y = self._data_generation(list_ids)

#				yield X, y


#	def _data_generation(self, list_ids):
#		#Generates data of batch_size samples' # X : (n_samples, v_size)
#		# Initialization
#		#X = np.empty((self.batch_size, self.dim_num, self.dim_state))
#		X = np.empty((self.batch_size, self.dim_state))
#		y = np.empty((self.batch_size), dtype = int)

#		for i in range(self.batch_size):
#			res = False
#			while not res:
#				sector = random.randint(0, len(list_ids)-1)
#				label  = random.randint(0, self.nclass - 1)
#				ndx    = random.randint(0, len(list_ids[sector][label]) - 1)

#				#res, data = load_and_sample_data(list_ids[sector][label][ndx], self.dim_num)
#				res, data = load_data_point(list_ids[sector][label][ndx])

#			X[i, :] = data
#			y[i] = label

#		return X, sparsify(y)



#def sparsify(y):
#	# Returns labels in binary NumPy array'
#	n_classes = NUM_CLASSES
#	return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
#					for i in range(y.shape[0])])



##parameters
#paramsTrain={#'dim_num': 1000,
#			 'dim_state': 5,
#			 'batch_size':BATCH_SIZE,
#			 'batches_per_epoch':BATCHES_PER_EPOCH,
#			 'nclass':NUM_CLASSES,}


#paramsValid={#'dim_num': 1000,
#			 'dim_state': 5,
#			 'batch_size':BATCH_SIZE,
#			 'batches_per_epoch':80,
#			 'nclass':NUM_CLASSES,}


## Generators
#print('creating generators...')#, end='')
#training_generator   = DataGenerator(**paramsTrain).generate(partition['train'])
#validation_generator = DataGenerator(**paramsValid).generate(partition['valid'])
#print('done')

#print('begin training')
#print('--------------------')
#time.sleep(2)


## Train model on dataset
#history = model.fit_generator(generator = training_generator,
#							  steps_per_epoch = paramsTrain['batches_per_epoch'],
#							  validation_data = validation_generator,
#							  validation_steps = paramsValid['batches_per_epoch'],
#							  epochs=1000,
#							  verbose=2,
#							  callbacks=callbacks_list)


#print(history)












#train_set = [[] for i in range(NUM_CLASSES)]
#train_set_ex1 = [[] for i in range(NUM_CLASSES)]
#train_set_ex2 = [[] for i in range(NUM_CLASSES)]
#train_set_ex3 = [[] for i in range(NUM_CLASSES)]

#train_set = [[[] for i in range(NUM_CLASSES)] for i in range(NUM_TRAINING_PARTITIONS)]
#valid_set = [[] for i in range(NUM_CLASSES)]

#for filepath in list_paths:
#	with open(filepath, 'rb') as f:
#		data_batch = pickle.load(f)

#		for i in range(NUM_CLASSES):
#			if random.randrange(2):
#				p = random.randrange(NUM_TRAINING_PARTITIONS)
#				train_set[p][i].











