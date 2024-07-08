import gym
import panda_gym
import pybullet_data
import time
import pybullet as p
import numpy as np



env = gym.make('PandaReach-v3', render_mode="vr")

observation, info = env.reset()

while True:
    action = env.action_space.sample()  # random action
    action_shape = action.shape
    action_new = np.zeros(action_shape)

    observation, reward, terminated, truncated, info = env.step(action_new)

    # if terminated or truncated:
    #      bservation, info = env.reset()

env.close()
