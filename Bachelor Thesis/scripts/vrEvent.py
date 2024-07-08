import gymnasium as gym
import panda_gym
import pybullet_data
import time
import pybullet as p
import numpy as np



env = gym.make('PandaPush-v3', render_mode="vr")


threshold = 0.1

current_position =env.task.get_robot_position()
last_position = current_position
observation, info = env.reset()


whether_calibrated = False

while not whether_calibrated:
    dist_1, dist_2 = env.task.distance()
    if dist_1 <= threshold and dist_2 <= threshold:
        whether_calibrated = True
        break

print(whether_calibrated)



while whether_calibrated:
    last_position = current_position
    current_position =env.task.get_robot_position()
    if last_position is not None and current_position is not None:
        action = (current_position - last_position) * 20
        action_new = np.zeros(action.shape())
        observation, reward, terminated, truncated, info = env.step(action)





env.close()
