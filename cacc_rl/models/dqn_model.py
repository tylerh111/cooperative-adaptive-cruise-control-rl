
import random
import numpy as np

from collections import deque

from keras.models import Sequential
from keras.layers import Dense
#from keras.activations import sigmoid
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf


#PATH_TO_WEIGHTS = 'E:\\comp594\\Cooperative-Adaptive-CC-RL\\cacc_rl\\models\\dqn_model_weights\\'

class DQNAgent:
	def __init__(self, state_size, action_size, weigths_path = None, gamma = 0.95, epsilon = 1.0, epsilon_min = 0.01, epsilon_decay = 0.99, learning_rate = 0.001, memory_size = 2000):
		self.state_size	 = state_size
		self.action_size = action_size

		self.gamma			= gamma		# discount rate
		self.epsilon		= epsilon	# exploration rate
		self.epsilon_min	= epsilon_min
		self.epsilon_decay	= epsilon_decay
		self.learning_rate	= learning_rate
		
		self.memory = deque(maxlen=memory_size)

		self.model = self._build_model()
		self.target_model = self._build_model(weigths_path)
		self.update_target_model()

		return


	"""
	Huber loss for Q Learning
	References: https://en.wikipedia.org/wiki/Huber_loss
				https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
	"""
	def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
		error = y_true - y_pred
		cond  = K.abs(error) <= clip_delta

		squared_loss = 0.5 * K.square(error)
		quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

		return K.mean(tf.where(cond, squared_loss, quadratic_loss))







	def _build_model(self, weights_path = None):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(128, input_dim=self.state_size, activation=K.sigmoid))
		model.add(Dense(256, activation=K.sigmoid))
		model.add(Dense(128, activation=K.sigmoid))
		model.add(Dense(self.action_size, activation=K.softmax))

		if weights_path is not None:
			model.load_weights(weights_path)

		model.compile(loss=self._huber_loss,
					  optimizer=Adam(lr=self.learning_rate))

		return model





	def update_target_model(self):
		# copy weights from model to target_model
		self.target_model.set_weights(self.model.get_weights())






	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))






	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
			
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action





	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)

		for state, action, reward, next_state, done in minibatch:
			target = self.model.predict(state)
			if done:
				target[0][action] = reward
			else:
				# a = self.model.predict(next_state)[0]
				t = self.target_model.predict(next_state)[0]
				target[0][action] = reward + self.gamma * np.amax(t)
				# target[0][action] = reward + self.gamma * t[np.argmax(a)]

			self.model.fit(state, target, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		return
		





	def load(self, path):
		self.model.load_weights(path)

	def save(self, path):
		self.model.save_weights(path)
	


