from gym_cacc.vehicle import Vehicle

from gym.envs.registration import register

register(
	id='cacc-v0', 
	entry_point='gym_cacc.envs:StopAndGo',
)
