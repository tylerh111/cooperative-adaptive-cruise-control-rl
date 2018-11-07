
import math
import numpy as np
import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_cacc import Vehicle


class Adversary(gym.Env):
	
	metadata = {
		'render.modes':[],
		'video.frames_per_second':60
	}



	"""
	Adversary: __init__
	
	"""
	def __init__(self, vehicle = Vehicle(), target_headway = 2, init_velocity = 25):
		
		#custom variables
		assert target_headway > 0, "target_headway is not positive"
		self.target_headway = target_headway

		assert init_velocity > 0, "init_velocity is not positive"
		self.init_velocity = init_velocity

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
			0,								#r_pos
			0,								#r_vel
			-self.vehicle.top_acceleration,	#r_acc
		])
		
		high = np.array([
			np.inf,						#headway
			np.inf,						#delta headway
			self.vehicle.top_acceleration,	#f_acc
			np.inf,							#r_pos
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
		The interval [-1, 1] is divided into 21 intervals:	[-1, -0.90), ... [-0.20, 0), [-0.10, 0), [0], (0, 0.10], (0.10, 0.20], ... (0.90, 1]
			interval compressed to a single value:			[-1],        ... [-0.20],    [-0.10],    [0],    [0.10],       [0.20], ...       [1]
			discrete value in action_space:					0,           ... 8,          9,          10,     11,           12,     ...       20
		'''
		self.action_space = spaces.Discrete(21) # jerk is a percentage interval from -1 to 1 (where 0 is no jerk)
		self.observation_space = spaces.Box(low, high, dtype=np.float32)

		self.seed()
		self.viewer = None

		self.state = None
		

		#Environment Variables
		self.headway_lower_bound = 0
		self.headway_upper_bound = 100 #MUST CHANGE!!!!!!! (in terms of target_headway)

		self.headway = self.target_headway
		self.delta_headway = 0

		self.num_steps = 0

		#Front vehicle kinematics
		self.f_pos = self.target_headway * self.init_velocity + self.vehicle.length
		self.f_vel = self.init_velocity
		self.f_acc = 0
		self.f_jer = 0
		
		#Rear vehicle kinematics
		self.r_pos = 0
		self.r_vel = self.init_velocity
		self.r_acc = 0
		self.r_jer = 0


		self.reward_total_front = 0
		self.reward_total_rear  = 0

		self.history_hw  = []
		self.history_dhw = []

		self.avg_hw  = 0
		self.avg_dhw = 0

		self.std_hw  = 0
		self.std_dhw = 0
		

		print('environment created')
		return


	

	"""
	Adversary: step
	
	"""
	def	step(self, action):
		#unwrap action
		action_front, action_rear = action
		assert self.action_space.contains(action_front), "%r (%s) invalid"%(action_front, type(action_front))
		assert self.action_space.contains(action_rear),  "%r (%s) invalid"%(action_rear, type(action_rear))
		
		state = self.state
		curr_state_front, curr_state_rear = state

		curr_hw, curr_dhw, curr_r_acc, curr_f_pos, curr_f_vel, curr_f_acc = curr_state_front
		curr_hw, curr_dhw, curr_f_acc, curr_r_pos, curr_r_vel, curr_r_acc = curr_state_rear

		#update front vehicle
		self.f_jer = (action_front - 10) / 10 * self.vehicle.top_jerk
		self.f_acc = max(-self.vehicle.top_acceleration, min(self.vehicle.top_acceleration,	self.f_jer / self.metadata['video.frames_per_second'] + self.f_acc))
		self.f_vel = max(0,								 min(self.vehicle.top_velocity,		self.f_acc / self.metadata['video.frames_per_second'] + self.f_vel))
		self.f_pos = self.f_vel / self.metadata['video.frames_per_second'] + self.f_pos


		
		#update rear vehicle
		self.r_jer = (action_rear - 10) / 10 * self.vehicle.top_jerk
		self.r_acc = max(-self.vehicle.top_acceleration, min(self.vehicle.top_acceleration,	self.r_jer / self.metadata['video.frames_per_second'] + self.r_acc))
		self.r_vel = max(0,								 min(self.vehicle.top_velocity,		self.r_acc / self.metadata['video.frames_per_second'] + self.r_vel))
		self.r_pos = self.r_vel / self.metadata['video.frames_per_second'] + self.r_pos
		

		#update other environment variables
		#self.headway = max(self.headway_lower_bound, np.inf if self.r_vel == 0 else (self.f_pos - self.vehicle.length - self.r_pos) / self.r_vel)
		self.headway = np.inf if self.r_vel == 0 else (self.f_pos - self.vehicle.length - self.r_pos) / self.r_vel
		self.delta_headway = self.headway - curr_hw

		#update state
		state_front = (self.headway, self.delta_headway, self.r_acc, self.f_pos, self.f_vel, self.f_acc)
		state_rear  = (self.headway, self.delta_headway, self.f_acc, self.r_pos, self.r_vel, self.r_acc)

		self.state = (state_front, state_rear)
		
		self.num_steps += 1


		self.history_hw.append(self.headway)
		self.history_dhw.append(self.delta_headway)

		def calcAvg(x):
			return sum(x) / len(x)

		def calcStdDev(x, m):
			c = 0
			for i in x:
				c += (i - m) ** 2

			return math.sqrt(c / len(x))


		self.avg_hw  = calcAvg(self.history_hw)
		self.avg_dhw = calcAvg(self.history_dhw)

		self.std_hw  = calcStdDev(self.history_hw,  self.avg_hw)
		self.std_dhw = calcStdDev(self.history_dhw, self.avg_hw)

		
		#decide if simulation is complete
		#done = too close (i.e. crash) OR too far
		#done = self.headway <= self.headway_lower_bound or \
		#	   self.headway >= self.headway_upper_bound 
		
		done = self.f_pos - self.r_pos <= 0 or \
			   self.f_pos - self.r_pos > self.headway_upper_bound * self.vehicle.top_velocity
		
		done = bool(done)

		def checkBounds(x, low, high):
			return low <= x and x <= high
		


		t_hw = self.target_headway


		if not done:
			if checkBounds(abs(self.avg_hw - t_hw), 0, 0.50)        and checkBounds(self.std_hw, 0, 0.50):
				reward_front = -50
				reward_rear  = 50
			elif checkBounds(abs(self.avg_hw - t_hw), 0.50, 1.25)   and checkBounds(self.std_hw, 0, 0.50):
				reward_front = -10
				reward_rear  = 10
			elif checkBounds(abs(self.avg_hw - t_hw), 1.25, 3.00)   and checkBounds(self.std_hw, 0, 0.50):
				reward_front = -5
				reward_rear  = 5
			elif checkBounds(abs(self.avg_hw - t_hw), 3.00, np.inf) and checkBounds(self.std_hw, 0, 0.50):
				reward_front = 50
				reward_rear  = -50
			elif checkBounds(abs(self.avg_hw - t_hw), 0, 0.50)      and checkBounds(self.std_hw, 0.50, 1.25):
				reward_front = -50
				reward_rear  = 50
			elif checkBounds(abs(self.avg_hw - t_hw), 0.50, 1.25)   and checkBounds(self.std_hw, 0.50, 1.25):
				reward_front = -10
				reward_rear  = 10
			elif checkBounds(abs(self.avg_hw - t_hw), 1.25, 3.00)   and checkBounds(self.std_hw, 0.50, 1.25):
				reward_front = -5
				reward_rear  = 5
			elif checkBounds(abs(self.avg_hw - t_hw), 3.00, np.inf) and checkBounds(self.std_hw, 0.50, 1.25):
				reward_front = 50
				reward_rear  = -50
			elif checkBounds(abs(self.avg_hw - t_hw), 0, 0.50)      and checkBounds(self.std_hw, 1.25, 3.00):
				reward_front = -50
				reward_rear  = 50
			elif checkBounds(abs(self.avg_hw - t_hw), 0.50, 1.25)   and checkBounds(self.std_hw, 1.25, 3.00):
				reward_front = -10
				reward_rear  = 10
			elif checkBounds(abs(self.avg_hw - t_hw), 1.25, 3.00)   and checkBounds(self.std_hw, 1.25, 3.00):
				reward_front = -5
				reward_rear  = 5
			elif checkBounds(abs(self.avg_hw - t_hw), 3.00, np.inf) and checkBounds(self.std_hw, 1.25, 3.00):
				reward_front = 50
				reward_rear  = -50
			elif checkBounds(abs(self.avg_hw - t_hw), 0, 0.50)      and checkBounds(self.std_hw, 3.00, np.inf):
				reward_front = -50
				reward_rear  = 50
			elif checkBounds(abs(self.avg_hw - t_hw), 0.50, 1.25)   and checkBounds(self.std_hw, 3.00, np.inf):
				reward_front = -10
				reward_rear  = 10
			elif checkBounds(abs(self.avg_hw - t_hw), 1.25, 3.00)   and checkBounds(self.std_hw, 3.00, np.inf):
				reward_front = -5
				reward_rear  = 5
			elif checkBounds(abs(self.avg_hw - t_hw), 3.00, np.inf) and checkBounds(self.std_hw, 3.00, np.inf):
				reward_front = 50
				reward_rear  = -50
		else:
			reward_rear = -100
			reward_front = 100


		
		##reward
		#if not done:
		#	#independent of delat_headway
		#	if checkBounds(self.headway, self.target_headway - 0.10, self.target_headway + 0.10):
		#		reward_rear = 50 #goal range small
		#		reward_front = -50
		#	elif checkBounds(self.headway, self.target_headway + 0.10, self.target_headway + 0.25):
		#		reward_rear = 5  #goal range large (far)
		#		reward_front = -5
		#	elif checkBounds(self.headway, self.target_headway - 0.25, self.target_headway - 0.10):
		#		reward_rear = 5  #goal range large (close)
		#		reward_front = -5

		#	#dependent of delta_headway and too close
		#	elif checkBounds(self.headway, self.target_headway - 1.50, self.target_headway - 0.25) and self.delta_headway <= 0: #checkBounds(self.delta_headway, -0.10, 0):
		#		reward_rear = -5 #too close
		#		reward_front = 5
		#	elif checkBounds(self.headway, self.target_headway - 1.50, self.target_headway - 0.25) and self.delta_headway > 0: #checkBounds(self.delta_headway, 0, 0.10):
		#		reward_rear = 1  #falling behind from being too close
		#		reward_front = -1
		#	elif checkBounds(self.headway, 0, self.target_headway - 1.50):
		#		reward_rear = -100 #crash
		#		reward_front = 100

		#	#dependent of delta_headway and too far
		#	elif checkBounds(self.headway, self.target_headway + 0.25, self.target_headway + 8.00) and self.delta_headway <= 0: #checkBounds(self.delta_headway, -0.10, 0):
		#		reward_rear = 1  #closing in from being too far
		#		reward_front = -1
		#	elif checkBounds(self.headway, self.target_headway + 0.25, self.target_headway + 8.00) and self.delta_headway > 0: #checkBounds(self.delta_headway, 0, 0.10):
		#		reward_rear = -1 #too far (and getting farther away)
		#		reward_front = 1
		#	elif checkBounds(self.headway, self.target_headway + 8.00, self.headway_upper_bound) and self.delta_headway <= 0: #checkBounds(self.delta_headway, -0.10, 0):
		#		reward_rear = 0.5 #closing in from being too far (even farther away)
		#		reward_front = -0.5 #0.5
		#	elif checkBounds(self.headway, self.target_headway + 8.00, self.headway_upper_bound) and self.delta_headway > 0: #checkBounds(self.delta_headway, 0, 0.10):
		#		reward_rear = -50 #way too far
		#		reward_front = 50
			
		#else:
		#	reward_rear = -100
		#	reward_front = 100
		
		
		#reward 
		#MUST UPDATE WITH MORE DETAIL!!!
		#if not done:
		#	reward_front = -1.0
		#	reward_rear  = 1.0
		#else:
		#	#logger.warn(" You called 'step()' after simulation was complete")
		#	reward_front = 1.0
		#	reward_rear  = -1.0

		return  (np.array(state_front), np.array(state_rear)), \
				(reward_front, reward_rear), \
				done, {}




	"""
	Adversary: reset
	
	"""
	def reset(self, vehicle = None, target_headway = None):

		#Environment Variables
		if vehicle != None:
			#assert vehicle > 0, "vehicle is not positive"
			self.vehicle = vehicle

		if target_headway != None:
			assert target_headway > 0, "target_headway is not positive"
			self.target_headway = target_headway

		self.headway = self.target_headway
		self.delta_headway = 0

		self.num_steps = 0

		#Front vehicle kinematics
		self.f_pos = self.target_headway * self.init_velocity + self.vehicle.length
		self.f_vel = self.init_velocity
		self.f_acc = 0
		self.f_jer = 0
		
		#Rear vehicle kinematics
		self.r_pos = 0
		self.r_vel = self.init_velocity
		self.r_acc = 0
		self.r_jer = 0

		self.reward_total_front = 0
		self.reward_total_rear  = 0

		self.history_hw  = []
		self.history_dhw = []


		
		#state
		state_front = (self.headway, self.delta_headway, self.r_acc, self.f_pos, self.f_vel, self.f_acc)
		state_rear  = (self.headway, self.delta_headway, self.f_acc, self.r_pos, self.r_vel, self.r_acc)

		self.state = (state_front, state_rear)

		#print('environment reset')
		return (np.array(state_front), np.array(state_rear))






	def variablesKinematicsFrontVehicle(self):
		return [round(self.f_pos, 4), round(self.f_vel, 4), round(self.f_acc, 4), round(self.f_jer, 4)]

	def variablesKinematicsRearVehicle(self):
		return [round(self.r_pos, 4), round(self.r_vel, 4), round(self.r_acc, 4), round(self.r_jer, 4)]

	def variablesEnvironment(self):
		return [[round(self.headway, 4), round(self.delta_headway, 4)], [round(self.avg_hw, 4), round(self.avg_dhw, 4)], [round(self.std_hw, 4), round(self.std_dhw, 4)]]

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]


	#def render(self, mode='human', close=False):
	#	pass
		

	def close(self):
		print('closing environment')
		return


