# Cooperative-Adaptive-CC-RL
CACC using a reiforcement learning approach

Imports for interactive python:
import numpy as np; import sys; import os; sys.path.append('E:\\comp594\\Cooperative-Adaptive-CC-RL\\cacc_rl')
import gym; from gym_cacc import Vehicle; from gym_cacc.envs import StopAndGo; from models.dqn_model import DQNAgent

This repository has absolute paths for saving simulation data for training. These need to be update for relative paths or options, though I do not have time to do that.

This project is a simulation of two cars that "implement" Cooperative Adaptive Cruise Control (CACC). The project is split up to contain simulation code and training code.
