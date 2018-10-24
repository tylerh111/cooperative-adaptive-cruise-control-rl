
import math
import numpy as np
import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_cacc import Vehicle

class StopAndGo(gym.Env):
	metadata = {
		'render.modes':[],
		'video.frames_per_second':60
	}

	#target_headway : meters
	#time_limit : seconds
	def __init__(self, vehicle = Vehicle(), target_headway = 2):
		
		self.vehicle = vehicle

		'''
		Observation State has the form (headway, delta headway, f_acc, r_vel, r_acc)
		ndx | name           | low       -> high
		0   | headway (hw)   | -inf      -> inf
		1   | delta hw (dhw) | -inf      -> inf
		2   | f_acc          | -top_acc. -> top_acc.
		3   | r_vel          | 0         -> top_velocity
		4   | r_acc          | -top_acc. -> top_acc.
		'''
		low = np.array([
			-np.inf,					#headway
			-np.inf,					#delta headway
			-self.vehicle.top_acceleration,	#f_acc
			0,							#r_vel
			-self.vehicle.top_acceleration,	#r_acc
		])
		
		high = np.array([
			np.inf,						#headway
			np.inf,						#delta headway
			self.vehicle.top_acceleration,	#f_acc
			self.vehicle.top_velocity,		#r_vel
			self.vehicle.top_acceleration,	#r_acc
		])

		'''
		Action Space has the form (jerk) #jerk refers to the derivative of the rear vehicle's acceleration
			 |     < 0         = 0        > 0
		jerk |  decelerate   nothing   accelerate
			 |   (brake)    (nothing)    (gas)
		'''
		#action_high = np.array([vehicle.top_jerk])

		#self.action_space = spaces.Box(-action_high, action_high , dtype=np.float32)

		'''
		Action Space is percentage of jerk [-1, 1] # jerk refers to the derivative of the rear vehicle's acceleration wrt time
			 |  -1 <= j < 0    j = 0    0 < j <= 1
		jerk |   decelerate   nothing   accelerate
			 |    (brake)    (nothing)    (gas)
		The interval [-1, 1] is divided into 21 intervals:	[-1, -0.90), ... [-0.20, 0), [0.10, 0), [0], (0, 0.10], (0.10, 0.20], ... (0.90, 1]
			interval compressed to a single value:			[-1],        ... [-0.20],    [-0.10],   [0],    [0.10],       [0.20], ...       [1]
			discrete value in action_space:					0,           ... 8,          9,         10,     11,           12,     ...       20
		'''
		self.action_space = spaces.Discrete(21) # jerk is a percentage interval from -1 to 1 (where 0 is no jerk)
		self.observation_space = spaces.Box(low, high, dtype=np.float32)

		self.seed()
		self.viewer = None

		self.state = None
		

		#Environment Variables
		assert target_headway > 0, "target_headway is not positive"
		self.target_headway = target_headway

		self.headway_lower_bound = 0
		self.headway_upper_bound = 100 #MUST CHANGE!!!!!!! (in terms of target_headway)

		self.headway = self.target_headway
		self.delta_headway = self.target_headway

		self.num_steps = 0

		#Front vehicle kinematics
		self.f_pos = self.target_headway + self.vehicle.length
		self.f_vel = 0
		self.f_acc = 0
		self.f_jer = 0
		
		#Rear vehicle kinematics
		self.r_pos = 0
		self.r_vel = 0
		self.r_acc = 0
		self.r_jer = 0
		
		print('environment created')
		return





	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
		


	def	step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		print('action = ', action)

		state = self.state
		curr_hw, curr_dhw, curr_f_acc, curr_r_vel, curr_r_acc = state

		#update front vehicle
		#MUST UPDATE WITH MORE DETAIL!!! (temporarily mimics rear vehicle)
		self.f_jer = (action - 10) / 10 * self.vehicle.top_jerk
		self.f_acc = min(self.vehicle.top_acceleration, self.f_jer + self.f_acc)
		self.f_vel = min(self.vehicle.top_velocity, self.f_acc + self.f_vel)
		self.f_pos = self.f_vel + self.f_pos
		

		#update rear vehicle
		#self.r_jer = min(self.vehicle.top_jerk, action[0])
		self.r_jer = (action - 10) / 10 * self.vehicle.top_jerk
		self.r_acc = min(self.vehicle.top_acceleration, self.r_jer + self.r_acc)
		self.r_vel = min(self.vehicle.top_velocity, self.r_acc + self.r_vel)
		self.r_pos = self.r_vel + self.r_pos
		
		#update other environment variables
		self.headway = self.f_pos - self.vehicle.length - self.r_pos
		self.delta_headway = self.headway - curr_hw

		#update state
		self.state = (self.headway, self.delta_headway, self.f_acc, self.r_vel, self.r_acc)
		
		self.num_steps += 1
		
		#decide if simulation is complete
		#done = too close (i.e. crash) OR too far
		done = self.headway <= self.headway_lower_bound or \
			   self.headway >= self.headway_upper_bound 
		done = bool(done)

		#reward 
		#MUST UPDATE WITH MORE DETAIL!!!
		if not done:
			reward = 1.0
		else:
			logger.warn("You called 'step()' after simulation was complete")
			reward = 0.0
		

		return np.array(self.state), reward, done, {}
		



	def reset(self, vehicle = None, target_headway = None):

		#Environment Variables
		if vehicle != None:
			#assert vehicle > 0, "vehicle is not positive"
			self.vehicle = vehicle

		if target_headway != None:
			assert target_headway > 0, "target_headway is not positive"
			self.target_headway = target_headway

		self.headway = self.target_headway
		self.delta_headway = self.target_headway

		self.num_steps = 0

		#Front vehicle kinematics
		self.f_pos = self.target_headway + self.vehicle.length
		self.f_vel = 0
		self.f_acc = 0
		self.f_jer = 0
		
		#Rear vehicle kinematics
		self.r_pos = 0
		self.r_vel = 0
		self.r_acc = 0
		self.r_jer = 0
		
		#state
		self.state = (self.headway, self.delta_headway, self.f_acc, self.r_vel, self.r_acc)

		print('environment reset')
		return np.array(self.state)


	#def render(self, mode='human', close=False):
	#	pass
		

	def close(self):
		print('closing environment')
		return


