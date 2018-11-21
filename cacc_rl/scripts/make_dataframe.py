
import pickle

print('opening file')
with open('E:\\comp594\\Cooperative-Adaptive-CC-RL\\cacc_rl\\env_state_mem\\stopandgo\\test.txt', 'rb') as fp:
	data = pickle.load(fp)

print('env_state_mem.txt openned')


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def getDF(e):
	data_front	= []
	data_rear	= []
	data_hw		= []
	data_avg_hw = []
	data_std_hw = []

	for d in data[e]:
		data_front.append(d[0])
		data_rear.append(d[1])
		data_hw.append(d[2][0])
		data_avg_hw.append(d[2][1])
		data_std_hw.append(d[2][2])

	df_front = pd.DataFrame(data_front, columns=['f_pos', 'f_vel', 'f_acc', 'f_jer'])
	df_rear  = pd.DataFrame(data_rear,  columns=['r_pos', 'r_vel', 'r_acc', 'r_jer'])


	df_hw = pd.DataFrame(data_hw, columns=['hw','dhw'])
	df_avg_hw = pd.DataFrame(data_avg_hw, columns=['avg_hw','avg_dhw'])
	df_std_hw = pd.DataFrame(data_std_hw, columns=['std_hw','std_dhw'])

	return (df_front, df_rear, df_hw, df_avg_hw, df_std_hw)



df_front, df_rear, df_hw, df_avg_hw, df_std_hw = getDF(0)


