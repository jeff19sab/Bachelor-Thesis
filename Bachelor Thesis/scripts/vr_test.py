import gymnasium as gym
import panda_gym
import pybullet_data
import time
import pybullet as p
import numpy as np
import keyboard
import threading
import json

# Flag to stop the calibration
stop_flag = False

# Define the key press listener function
def on_press(event):
    global stop_flag
    if event.name == 'q':
        stop_flag = True


data_dict = {"task_2": {}}

# for i in range(5):
#     keyboard.on_press(on_press)
#     env = gym.make('PandaReach-v3', render_mode="vr")
#
#     threshold = 0.1
#
#     current_position =env.task.get_robot_position()
#     last_position = current_position
#     observation, info = env.reset()
#
#
#     whether_calibrated = False
#
#     while not whether_calibrated:
#         dist_1, dist_2 = env.task.distance()
#         if dist_1 <= threshold and dist_2 <= threshold:
#             whether_calibrated = True
#             break
#
#     print(whether_calibrated)
#     action_list = []
#     while whether_calibrated:
#         target = env.task.target()
#         last_position = current_position
#         current_position =env.task.get_robot_position()
#         if last_position is not None and current_position is not None:
#             gripper_width, gripper_width_2 = env.task.get_robot_gripper()
#             action = (current_position - last_position) * 20
#             action[3] = gripper_width
#             action[7] = gripper_width_2
#             action_list.append(list(action))
#             observation, reward, terminated, truncated, info = env.step(action)
#             if target or stop_flag:
#                 if target:
#                     success = True
#                 if stop_flag:
#                     success = False
#                 break
#     stop_flag = False
#     data_dict["task_2"][f'trial_{i+1}'] = {'actions': action_list, 'success': success}
#     # Save the dictionary to a file
#
#     env.close()
#     i+= 1

#second condition

for i in range(5):
    keyboard.on_press(on_press)
    env = gym.make('PandaReach-v3', render_mode="vr")

    threshold = 0.1

    current_position =env.task.handler_position_action()
    last_position = current_position
    observation, info = env.reset()


    whether_calibrated = False

    while not whether_calibrated:
        dist_1, dist_2 = env.task.distance()
        if dist_1 <= threshold and dist_2 <= threshold:
            whether_calibrated = True
            break

    print(whether_calibrated)
    action_list = []
    while whether_calibrated:
        target = env.task.target()
        last_position = current_position
        current_position =env.task.handler_position_action()
        if last_position is not None and current_position is not None:
            gripper_width, gripper_width_2 = env.task.get_robot_gripper()
            action = (current_position - last_position) * 20
            action[3] = gripper_width
            action[7] = gripper_width_2
            action_list.append(list(action))
            observation, reward, terminated, truncated, info = env.step(action)
            if target or stop_flag:
                if target:
                    success = True
                if stop_flag:
                    success = False
                break
    stop_flag = False
    data_dict["task_2"][f'trial_{i+1}'] = {'actions': action_list, 'success': success}
    # Save the dictionary to a file

    env.close()
    i+= 1

with open('C:/Users/Jeffs/Desktop/experiments/ID 18/task_data_task_2.json', 'w') as json_file:
    json.dump(data_dict, json_file, indent=4)