
import tensorflow as tf

#import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import LSTM, CuDNNLSTM, Dense, Input, Reshape, Flatten, Dropout, BatchNormalization, GlobalMaxPool2D, Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers, losses, activations, models, metrics, utils
from tensorflow.keras import backend as K


def build_model(input_shape, output_shape, learning_rate = 1e-2, learning_rate_decay = 1e-6, weights_path = None):

	model = Sequential()

	model.add(LSTM(128, input_shape=input_shape, activation=K.sigmoid, return_sequences=True, name='lstm_fc1'))
	model.add(Dropout(0.2, name='dropout_fc1'))
	model.add(LSTM(128, activation=K.sigmoid, name='lstm_fc2'))
	model.add(Dropout(0.2, name='dropout_fc2'))
	model.add(Dense(32, activation=K.sigmoid, name='dense_fc3'))
	model.add(Dropout(0.2, name='dropout_fc3'))
	model.add(Dense(output_shape, activation=K.softmax, name='actions'))
	
	if weights_path is not None:
		model.load_weights(weights_path)

	opt = Adam(lr=learning_rate, decay=learning_rate_decay)
	#opt = SGD(lr=LEARNING_RATE, decay=LEARNING_RATE_DECAY, momentum=LEARNING_RATE_MOMENTUM)

	model.compile(loss='categorical_crossentropy',
					optimizer=opt,
					metrics=['accuracy'])

	return model

