
import math
import numpy as np
import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_cacc import Vehicle


fps = 60


class Adversary(gym.Env):
	
	metadata = {
		'render.modes':[],
		'video.frames_per_second':60
	}



	"""
	Adversary: __init__
	
	"""
	def __init__(self,):

		self.seed = self.seed_env()

		self.vehicle = None
		self.granularity = None
		self.init_vel = None
		self.init_acc = None
		
		self.action_space = None
		self.observation_space = None

		self.state = None
		

		#Environment Variables
		self.headway_lower_bound = 0
		self.headway_upper_bound = 0

		self.hw = 0
		self.dhw = 0
		self.target_hw = 0

		self.num_steps = 0

		#Front vehicle kinematics
		self.f_pos = 0
		self.f_vel = 0
		self.f_acc = 0
		#self.f_jer = 0
		
		#Rear vehicle kinematics
		self.r_pos = 0
		self.r_vel = 0
		self.r_acc = 0
		#self.r_jer = 0


		self.reward_total_front = 0
		self.reward_total_rear  = 0

		self.history_hw  = []
		self.history_dhw = []

		self.avg_hw  = 0
		self.avg_dhw = 0

		self.std_hw  = 0
		self.std_dhw = 0
		

		return



	def _update_action_space(self, granularity):

		'''
		Action Space is percentage of jerk [-1, 1] # jerk refers to the derivative of the rear vehicle's acceleration wrt time
			 |  -1 <= j < 0    j = 0    0 < j <= 1
		jerk |   decelerate   nothing   accelerate
			 |    (brake)    (nothing)    (gas)
		The interval [-1, 1] is divided into 21 intervals:	[-1, -0.90), ... [-0.20, 0), [-0.10, 0), [0], (0, 0.10], (0.10, 0.20], ... (0.90, 1]
			interval compressed to a single value:			[-1],        ... [-0.20],    [-0.10],    [0],    [0.10],       [0.20], ...       [1]
			discrete value in action_space:					0,           ... 8,          9,          10,     11,           12,     ...       20
		'''

		# jerk is a percentage interval from -1 to 1 (where 0 is no jerk)
		
		self.action_space = spaces.Discrete(granularity) 

	def _update_observation_space(self, vehicle):

		'''
		Observation State has the form (headway, delta headway, f_acc, r_vel, r_acc)
		ndx | name           | low       -> high
		0   | headway (hw)   | -inf      -> inf
		1   | delta hw (dhw) | -inf      -> inf
		2   | f_acc          | -top_acc. -> top_acc.
		3   | r_vel          | 0         -> top_velocity
		4   | r_acc          | -top_acc. -> top_acc.
		5   | r_jer          | -top_jer. -> top_jer.
		'''
		low = np.array([
			-np.inf,					#headway
			-np.inf,					#delta headway
			-vehicle.top_acc,	#f_acc
			0,							#r_vel
			-vehicle.top_acc,	#r_acc
			#-vehicle.top_jer,
		])
		
		high = np.array([
			np.inf,						#headway
			np.inf,						#delta headway
			vehicle.top_acc,	#f_acc
			vehicle.top_vel,		#r_vel
			vehicle.top_acc,	#r_acc
			#vehicle.top_jer,
		])

		self.observation_space = spaces.Box(low, high, dtype=np.float32)



	
	"""
	Adversary: reward function
	
	"""
	def _reward_function(self, t_hw, avg_hw, std_hw):

		def checkBounds(x, low, high):
			return low <= x and x <= high

		if self.avg_hw == np.inf or self.std_hw == np.inf:
			reward = -100
			bound  = 'inf'
		#---------------------------------------------------------------------------------------------------
		elif checkBounds(abs(self.avg_hw - t_hw), 0, 0.50)      and checkBounds(self.std_hw, 0, 0.50):
			reward = 50
			bound  = 'AV'
		elif checkBounds(abs(self.avg_hw - t_hw), 0.50, 1.25)   and checkBounds(self.std_hw, 0, 0.50):
			reward = 5
			bound  = 'BV'
		elif checkBounds(abs(self.avg_hw - t_hw), 1.25, 1.50)   and checkBounds(self.std_hw, 0, 0.50):
			reward = 1
			bound  = 'CV'
		elif checkBounds(abs(self.avg_hw - t_hw), 1.50, 3.00)   and checkBounds(self.std_hw, 0, 0.50):
			reward = -5
			bound  = 'DV'
		elif checkBounds(abs(self.avg_hw - t_hw), 3.00, np.inf) and checkBounds(self.std_hw, 0, 0.50):
			reward = -50
			bound  = 'EV'
		#---------------------------------------------------------------------------------------------------
		elif checkBounds(abs(self.avg_hw - t_hw), 0, 0.50)      and checkBounds(self.std_hw, 0.50, 1.25):
			reward = 5
			bound  = 'AW'
		elif checkBounds(abs(self.avg_hw - t_hw), 0.50, 1.25)   and checkBounds(self.std_hw, 0.50, 1.25):
			reward = 1
			bound  = 'BW'
		elif checkBounds(abs(self.avg_hw - t_hw), 1.25, 1.50)   and checkBounds(self.std_hw, 0.50, 1.25):
			reward = 1
			bound  = 'CW'
		elif checkBounds(abs(self.avg_hw - t_hw), 1.50, 3.00)   and checkBounds(self.std_hw, 0.50, 1.25):
			reward = -5
			bound  = 'DW'
		elif checkBounds(abs(self.avg_hw - t_hw), 3.00, np.inf) and checkBounds(self.std_hw, 0.50, 1.25):
			reward = -50
			bound  = 'EW'
		#---------------------------------------------------------------------------------------------------
		elif checkBounds(abs(self.avg_hw - t_hw), 0, 0.50)      and checkBounds(self.std_hw, 1.25, 1.50):
			reward = 1
			bound  = 'AX'
		elif checkBounds(abs(self.avg_hw - t_hw), 0.50, 1.25)   and checkBounds(self.std_hw, 1.25, 1.50):
			reward = 1
			bound  = 'BX'
		elif checkBounds(abs(self.avg_hw - t_hw), 1.25, 1.50)   and checkBounds(self.std_hw, 1.25, 1.50):
			reward = -1
			bound  = 'CX'
		elif checkBounds(abs(self.avg_hw - t_hw), 1.50, 3.00)   and checkBounds(self.std_hw, 1.25, 1.50):
			reward = -5
			bound  = 'DX'
		elif checkBounds(abs(self.avg_hw - t_hw), 3.00, np.inf) and checkBounds(self.std_hw, 1.25, 1.50):
			reward = -50
			bound  = 'EX'
		#---------------------------------------------------------------------------------------------------
		elif checkBounds(abs(self.avg_hw - t_hw), 0, 0.50)      and checkBounds(self.std_hw, 1.50, 3.00):
			reward = -5
			bound  = 'AY'
		elif checkBounds(abs(self.avg_hw - t_hw), 0.50, 1.25)   and checkBounds(self.std_hw, 1.50, 3.00):
			reward = -5
			bound  = 'BY'
		elif checkBounds(abs(self.avg_hw - t_hw), 1.25, 1.50)   and checkBounds(self.std_hw, 1.50, 3.00):
			reward = -5
			bound  = 'CY'
		elif checkBounds(abs(self.avg_hw - t_hw), 1.50, 3.00)   and checkBounds(self.std_hw, 1.50, 3.00):
			reward = -10
			bound  = 'DY'
		elif checkBounds(abs(self.avg_hw - t_hw), 3.00, np.inf) and checkBounds(self.std_hw, 1.50, 3.00):
			reward = -55
			bound  = 'EY'
		#---------------------------------------------------------------------------------------------------
		elif checkBounds(abs(self.avg_hw - t_hw), 0, 0.50)      and checkBounds(self.std_hw, 3.00, np.inf):
			reward = -10
			bound  = 'AZ'
		elif checkBounds(abs(self.avg_hw - t_hw), 0.50, 1.25)   and checkBounds(self.std_hw, 3.00, np.inf):
			reward = -10
			bound  = 'BZ'
		elif checkBounds(abs(self.avg_hw - t_hw), 1.25, 1.50)   and checkBounds(self.std_hw, 3.00, np.inf):
			reward = -10
			bound  = 'CZ'
		elif checkBounds(abs(self.avg_hw - t_hw), 1.50, 3.00)   and checkBounds(self.std_hw, 3.00, np.inf):
			reward = -20
			bound  = 'DZ'
		elif checkBounds(abs(self.avg_hw - t_hw), 3.00, np.inf) and checkBounds(self.std_hw, 3.00, np.inf):
			reward = -100
			bound  = 'EZ'

		return (reward, bound)


	"""
	Adversary: reward function
	
	"""
	def _reward_function_old(self, t_hw, hw, dhw):

		def checkBounds(x, low, high):
			return low <= x and x <= high

		#independent of delat_headway
		if checkBounds(hw, t_hw - 0.10, t_hw + 0.10):
			reward = 50 #goal range small
			bound = 'A'
		elif checkBounds(hw, t_hw + 0.10, t_hw + 0.25):
			reward = 5  #goal range large (far)
			bound = 'B'
		elif checkBounds(hw, t_hw - 0.25, t_hw - 0.10):
			reward = 5  #goal range large (close)
			bound = 'C'

		#dependent of delta_headway and too close
		elif checkBounds(hw, t_hw - 1.50, t_hw - 0.25) and dhw <= 0: #checkBounds(dhw, -0.10, 0):
			reward = -5 #too close
			bound = 'D'
		elif checkBounds(hw, t_hw - 1.50, t_hw - 0.25) and dhw > 0: #checkBounds(dhw, 0, 0.10):
			reward = 1  #falling behind from being too close
			bound = 'E'
		elif checkBounds(hw, -np.inf, t_hw - 1.50):
			reward = -100 #crash
			bound = 'DONE??'

		#dependent of delta_headway and too far
		elif checkBounds(hw, t_hw + 0.25, t_hw + 8.00) and dhw <= 0: #checkBounds(dhw, -0.10, 0):
			reward = 1  #closing in from being too far
			bound = 'F'
		elif checkBounds(hw, t_hw + 0.25, t_hw + 8.00) and dhw > 0: #checkBounds(dhw, 0, 0.10):
			reward = -1 #too far (and getting farther away)
			bound = 'G'
		elif checkBounds(hw, t_hw + 8.00, np.inf) and dhw <= 0: #checkBounds(dhw, -0.10, 0):
			reward = 0.5 #closing in from being too far (even farther away)
			bound = 'H'
		elif checkBounds(hw, t_hw + 8.00, np.inf) and dhw > 0: #checkBounds(dhw, 0, 0.10):
			reward = -50 #way too far
			bound = 'I'
		elif hw == np.inf and dhw == np.nan:
			reward = -50
			bound = 'inf'
		
		return (reward, bound)



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

		curr_hw, curr_dhw, curr_r_acc, curr_f_vel, curr_f_acc = curr_state_front
		curr_hw, curr_dhw, curr_f_acc, curr_r_vel, curr_r_acc = curr_state_rear


		self.f_eq_point = (action_front - self.zero_point) / self.zero_point * self.vehicle.top_acc
		self.r_eq_point = (action_rear  - self.zero_point) / self.zero_point * self.vehicle.top_acc


		def getNextAcc(eq, a):
			if eq == 0:
				return 0
			if eq < 0:
				return max(eq, -self.vehicle.top_jer / fps + a)
			if eq > 0:
				return min(eq,  self.vehicle.top_jer / fps + a)
			raise ValueError('%r (%s) invalid'%(eq, type(eq)))
		
		#update front vehicle
		#self.f_jer = (action_front - 10) / 10 * self.vehicle.top_jer
		#self.f_acc = max(-self.vehicle.top_acc, min(self.vehicle.top_acc,	self.f_jer / fps + self.f_acc))
		self.f_acc = getNextAcc(self.f_eq_point, self.f_acc)
		self.f_vel = max(0, min(self.vehicle.top_vel, self.f_acc / fps + self.f_vel))
		self.f_pos = self.f_vel / fps + self.f_pos

		
		#update rear vehicle
		#self.r_jer = (action_rear - 10) / 10 * self.vehicle.top_jer
		#self.r_acc = max(-self.vehicle.top_acc, min(self.vehicle.top_acc,	self.r_jer / fps + self.r_acc))
		self.r_acc = getNextAcc(self.r_eq_point, self.r_acc)
		self.r_vel = max(0,	min(self.vehicle.top_vel, self.r_acc / fps + self.r_vel))
		self.r_pos = self.r_vel / fps + self.r_pos
		

		#update other environment variables
		#self.hw = max(self.headway_lower_bound, np.inf if self.r_vel == 0 else (self.f_pos - self.vehicle.length - self.r_pos) / self.r_vel)
		self.hw = (self.f_pos - self.vehicle.length - self.r_pos) / self.r_vel if self.r_vel != 0 else np.inf
		if self.hw != np.inf and curr_hw != np.inf:
			self.dhw = self.hw - curr_hw
		elif self.hw != np.inf and curr_hw == np.inf:
			self.dhw = self.hw
		else:
			self.dhw = self.hw - curr_hw
		
		#update state
		state_front = (self.hw, self.dhw, self.r_acc, self.f_vel, self.f_acc)
		state_rear  = (self.hw, self.dhw, self.f_acc, self.r_vel, self.r_acc)

		self.state = (state_front, state_rear)
		
		self.num_steps += 1


		self.history_hw.append(self.hw)
		self.history_dhw.append(self.dhw)



		def calcAvg(x):
			return np.mean(x)

		def calcStdDev(x):
			return np.std(x)
		
		self.avg_hw  = calcAvg(self.history_hw)
		self.avg_dhw = calcAvg(self.history_dhw)

		self.std_hw  = calcStdDev(self.history_hw)
		self.std_dhw = calcStdDev(self.history_dhw)

		
		#decide if simulation is complete
		#done = too close (i.e. crash) OR too far
		done = self.hw <= self.headway_lower_bound or \
			   self.hw >= self.headway_upper_bound 
		
		#done = self.f_pos - self.r_pos <= 0 or \
		#	   self.f_pos - self.r_pos > self.headway_upper_bound * self.vehicle.top_vel
		
		done = bool(done)

		if not done:
			#reward, bound = self._reward_function(self.target_hw, self.avg_hw, self.std_hw)
			try:
				reward, bound = self._reward_function_old(self.target_hw, self.hw, self.dhw)
			except:
				print('self.target_hw =', self.target_hw)
				print('self.hw        =', self.hw)
				print('self.dhw       =', self.dhw)
				raise

		else:
			reward = -150
			bound  = 'DONE'

		return  (np.array(state_front), np.array(state_rear)), \
				(-reward, reward), \
				done, \
				{'bound':bound}




	"""
	Adversary: reset
	
	"""
	def reset(self, vehicle = None, target_hw = None, granularity = None, init_vel = None, alpha_vel = None, init_acc = None, alpha_acc = None):

		self.seed = self.seed_env()

		#checking custom variable settings
		def checkCustomVariables():
			if vehicle != None:
				assert type(vehicle) is Vehicle, "vehicle is not type Vehicle"
				self.vehicle = vehicle
			else:
				self.vehicle = Vehicle()

			if target_hw != None:
				assert target_hw > 0, "target_hw is not positive"
				self.target_hw = target_hw
			else:
				self.target_hw = 2

			if granularity != None:
				assert granularity > 0, "granularity is not positive"
				assert granularity % 2 == 1, "granularity is not odd"
				self.granularity = granularity
			else:
				self.granularity = 21

			if init_vel != None:
				assert init_vel >= 0, "init_vel is negitive"
				assert init_vel <= self.vehicle.top_vel, "init_vel is higher than top_velocity"
				self.init_vel = init_vel
			else:
				self.init_vel = 0

			if alpha_vel != None:
				assert alpha_vel >= 0, "alpha_vel is negitive"
				assert alpha_vel <= self.vehicle.top_vel, "alpha_vel is higher than top_velocity"
				self.alpha_vel = alpha_vel
			else:
				self.alpha_vel = 0

			if init_acc != None:
				assert init_acc >= -self.vehicle.top_acc, "init_acc is lower than top_acceleration"
				assert init_acc <=  self.vehicle.top_acc, "init_acc is higher than top_acceleration"
				self.init_acc = init_acc
			else:
				self.init_acc = 0

			if alpha_acc != None:
				assert alpha_acc >= 0, "alpha_acc is negitive"
				assert alpha_acc <= self.vehicle.top_acc, "alpha_acc is higher than top_velocity"
				self.alpha_acc = alpha_acc
			else:
				self.alpha_acc = 0

		checkCustomVariables()


		self.zero_point = (self.granularity - 1) / 2
		
		self._update_action_space(self.granularity)
		self._update_observation_space(self.vehicle)

		self.headway_lower_bound = 0
		self.headway_upper_bound = 3 * self.target_hw

		self.num_steps = 0

		#Front vehicle kinematics
		self.f_pos = random.uniform(self.target_hw, self.target_hw + 1) * self.init_vel + self.vehicle.length
		self.f_vel = random.gauss(self.init_vel, self.alpha_vel)
		self.f_acc = random.gauss(self.init_acc, self.alpha_acc)
		#self.f_jer = 0

		self.f_eq_point = 0 #action of the front vehicle
		
		#Rear vehicle kinematics
		self.r_pos = 0
		self.r_vel = random.gauss(self.init_vel, self.alpha_vel)
		self.r_acc = random.gauss(self.init_acc, self.alpha_acc)
		#self.r_jer = 0
		
		self.r_eq_point = 0 #aciton of the rear vehicle


		self.reward_total_front = 0
		self.reward_total_rear  = 0

		self.hw = self.f_pos / self.r_vel
		self.dhw = 0

		self.history_hw  = [self.hw]
		self.history_dhw = [self.dhw]

		self.avg_hw  = self.hw
		self.avg_dhw = self.dhw

		self.std_hw  = self.hw
		self.std_dhw = self.dhw


		#state
		state_front = (self.hw, self.dhw, self.r_acc, self.f_vel, self.f_acc)
		state_rear  = (self.hw, self.dhw, self.f_acc, self.r_vel, self.r_acc)

		self.state = (state_front, state_rear)

		return (np.array(state_front), np.array(state_rear))

	

	def variablesKinematicsFrontVehicle(self):
		return [self.f_pos, self.f_vel, self.f_acc]#, self.f_jer]

	def variablesKinematicsRearVehicle(self):
		return [self.r_pos, self.r_vel, self.r_acc]#, self.r_jer]

	def variablesEnvironment(self):
		return [[self.hw, self.dhw], [self.avg_hw, self.avg_dhw], [self.std_hw, self.std_dhw]]

	def getSeed(self):
		return self.seed







	def seed_env(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

		

	#def close(self):
	#	return


