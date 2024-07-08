import os
import json
from collections import defaultdict
import numpy as np

def get_average_time_per_task(base_path):
    # Initialize dictionaries to store total time, count, and times for each task within each condition
    condition_task_times = {
        'Teleoperation': defaultdict(lambda: {'total_time': 0, 'count': 0, 'times': []}),
        'Egocentric': defaultdict(lambda: {'total_time': 0, 'count': 0, 'times': []}),
    }

    # Walk through the directory structure
    for condition in condition_task_times.keys():
        condition_path = os.path.join(base_path, condition)
        for participant in os.listdir(condition_path):
            participant_path = os.path.join(condition_path, participant)
            if os.path.isdir(participant_path):
                for file_name in os.listdir(participant_path):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(participant_path, file_name)
                        try:
                            with open(file_path, 'r') as file:
                                data_dict = json.load(file)
                                for task in ['task_1', 'task_2', 'task_3']:
                                    if task in data_dict:
                                        for trial in data_dict[task]:
                                            time = data_dict[task][trial]['time']
                                            condition_task_times[condition][task]['total_time'] += time
                                            condition_task_times[condition][task]['count'] += 1
                                            condition_task_times[condition][task]['times'].append(time)
                        except json.JSONDecodeError as e:
                            print(f"Error reading {file_path}: {e}")
                        except KeyError as e:
                            print(f"Missing expected key in {file_path}: {e}")

    # Calculate the average time and standard deviation for each task within each condition
    average_times_per_condition = {
        condition: {
            task: {
                'average_time': task_info['total_time'] / task_info['count'] if task_info['count'] > 0 else 0,
                'std_dev': np.std(task_info['times']) if task_info['count'] > 0 else 0
            }
            for task, task_info in tasks.items()
        }
        for condition, tasks in condition_task_times.items()
    }

    return average_times_per_condition

# Specify the path to the 'experiment' directory
base_path = 'C:/Users/Jeffs/Desktop/experiments'

# Get the average time per task
average_times = get_average_time_per_task(base_path)
import matplotlib.pyplot as plt
import numpy as np

# Data from the dictionary

# Create a figure and axis
fig, ax = plt.subplots()

# Define the tasks and conditions
tasks = ['task_1', 'task_2', 'task_3']
conditions = list(average_times.keys())

# Define bar width and positions
bar_width = 0.35
index = np.arange(len(tasks))

# Plot bars for each condition with error bars
for i, condition in enumerate(conditions):
    times = [average_times[condition][task]['average_time'] for task in tasks]
    errors = [average_times[condition][task]['std_dev'] for task in tasks]
    ax.bar(index + i * bar_width, times, bar_width, yerr=errors, label=condition, capsize=5)

# Add labels and title
ax.set_xlabel('Tasks')
ax.set_ylabel('Average Time(s)')
ax.set_title('Average Time per Task for Each Condition')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(tasks)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
