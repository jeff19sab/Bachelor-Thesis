import gymnasium as gym
import panda_gym
import pybullet_data
import time
import pybullet as p
import numpy as np

cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
    p.connect(p.GUI)
p.resetSimulation()

p.setAdditionalSearchPath(pybullet_data.getDataPath())
# disable rendering during loading makes it much faster
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

env = gym.make('PandaReach-v3', render_mode="vr")


observation, info = env.reset()

while True:
    action = env.action_space.sample() # random action
    action_shape = action.shape
    action_new = np.zeros(action_shape)
    observation, reward, terminated, truncated, info = env.step(action_new)

    # if terminated or truncated:
    #     observation, info = env.reset()

env.close()
