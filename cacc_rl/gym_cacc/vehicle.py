
class Vehicle():
	
	def __init__(self, length = 1, width = 1, height = 1, weight = 1, top_velocity = 1, top_acceleration = 1, top_jerk = 1, statistics = None):
		if statistics != None:
			assert type(statistics) is tuple, "statistics is not a tuple"
			assert len(statistics) == 7, "statistics is not a 7-tuple"
			length, width, height, weight, top_velocity, top_acceleration, top_jerk = statistics


		assert length > 0, "length is not positive"
		assert width > 0, "width is not positive"
		assert height > 0, "height is not positive"
		self.length = length
		self.width = width
		self.height = height

		assert weight > 0, "weight is not positive"
		self.weight = weight
		
		assert top_velocity > 0, "top_velocity is not positive"
		assert top_acceleration > 0, "top_acceleration is not positive"
		assert top_jerk > 0, "top_jerk is not positive"
		self.top_velocity = top_velocity
		self.top_acceleration = top_acceleration
		self.top_jerk = top_jerk

	def toArray(self):
		return [self.length, self.width, self.height, self.weight,
		  self.top_velocity, self.top_acceleration, self.top_jerk]

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return str(self.toArray())

		
