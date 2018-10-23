
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
	def __init__(self, vehicle = Vehicle(), target_headway = 2, time_limit = 15):
		
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
			-vehicle.top_acceleration,	#f_acc
			0,							#r_vel
			-vehicle.top_acceleration,	#r_acc
		])
		
		high = np.array([
			np.inf,						#headway
			np.inf,						#delta headway
			vehicle.top_acceleration,	#f_acc
			vehicle.top_velocity,		#r_vel
			vehicle.top_acceleration,	#r_acc
		])

		'''
		Action Space has the form (jerk) #jerk refers to the derivative of the rear vehicle's acceleration
			 |     < 0         = 0        > 0
		jerk |  decelerate   nothing   accelerate
			 |   (brake)    (nothing)    (gas)
		'''
		action_high = np.array([vehicle.top_jerk])

		self.action_space = spaces.Box(-action_high, action_high , dtype=np.float32)
		self.observation_space = spaces.Box(low, high, dtype=np.float32)

		self.seed()
		self.viewer = None

		self.state = None
		

		#Environment Variables
		assert target_headway > 0, "target_headway is not positive"
		self.target_headway = target_headway

		self.headway_lower_bound = 0
		self.headway_upper_bound = 100 #MUST CHANGE!!!!!!! (in terms of target_headway)

		self.headway = target_headway
		self.delta_headway = target_headway

		self.num_steps = 0
		self.time_limit = time_limit

		#Front vehicle kinematics
		self.f_pos = target_headway + vehicle.length
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
		#MUST UPDATE WITH MORE DETAIL!!!


		#update rear vehicle
		self.r_jer = min(self.vehicle.top_jerk, action[0])
		self.r_acc = min(self.vehicle.top_acceleration, self.r_jer + self.r_acc)
		self.r_vel = min(self.vehicle.top_velocity, self.r_acc + self.r_vel)
		self.r_pos = min(self.vehicle.top_jerk, self.r_vel + self.r_pos)
		
		#update other environment variables
		self.headway = self.f_pos - self.vehicle.length - self.r_pos
		self.delta_headway = self.headway - curr_hw

		#update state
		self.state = (self.headway, self.delta_headway, self.f_acc, self.r_vel, self.r_acc)
		
		#decide if simulation is complete
		self.num_steps += 1
		
		done = self.headway <= self.headway_lower_bound or \
			   self.headway >= self.headway_upper_bound or \
			   self.num_steps >= self.time_limit * metadata['video.frames_per_second'] 
		
		done = bool(done)

		#reward 
		#MUST UPDATE WITH MORE DETAIL!!!
		if not done:
			reward = 1.0
		else:
			logger.warn("You called 'step()' after simulation was complete")
			reward = 0.0
		

		return np.array(self.state), reward, done, {}
		



	def reset(self, target_headway = None, time_limit = None):

		#Environment Variables
		if target_headway != None:
			assert target_headway > 0, "target_headway is not positive"
			self.target_headway = target_headway

		if time_limit != None:
			assert time_limit > 0, "time_limit is not positive"
			self.time_limit = time_limit

		self.headway = self.target_headway
		self.delta_headway = self.target_headway

		self.num_steps = 0
		self.time_limit = self.time_limit

		#Front vehicle kinematics
		self.f_pos = target_headway + vehicle.length
		self.f_vel = 0
		self.f_acc = 0
		self.f_jer = 0
		
		#Rear vehicle kinematics
		self.r_pos = 0
		self.r_vel = 0
		self.r_acc = 0
		self.r_jer = 0
		
		print('environment reset')
		return


	#def render(self, mode='human', close=False):
	#	pass
		

	def close(self):
		print('closing environment')
		return


