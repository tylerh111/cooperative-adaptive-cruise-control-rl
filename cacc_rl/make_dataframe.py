
import pickle

print('opening file')
with open('B:\\comp594\\Cooperative-Adaptive-CC-RL\\cacc_rl\\env_state_mem.txt', 'rb') as fp:
	data = pickle.load(fp)

print('env_state_mem.txt openned')


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#using episode 1
e = 1

#create data lists
data_front = []

for d in data[e]:
	data_front.append(d[0])

data_rear = []
for d in data[e]:
	data_rear.append(d[1])

#create dataframe from data_front and data_rear

df_front = pd.DataFrame(data_front, columns=['pos', 'vel', 'acc', 'jer'])
df_rear  = pd.DataFrame(data_rear,  columns=['pos', 'vel', 'acc', 'jer'])


data_hw = []
for d in data[e]:
	data_hw.append(d[2][0])

data_avg_hw = []
for d in data[e]:
	data_avg_hw.append(d[2][1])

data_std_hw = []
for d in data[e]:
	data_std_hw.append(d[2][2])


df_hw = pd.DataFrame(data_hw, columns=['hw','dhw'])
df_avg_hw = pd.DataFrame(data_avg_hw, columns=['avg_hw','avg_dhw'])
df_std_hw = pd.DataFrame(data_std_hw, columns=['std_hw','std_dhw'])













