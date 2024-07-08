import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def get_average_time_per_trial_for_task_2(base_path):
    # Initialize dictionaries to store times for each trial in task_2 for each condition
    condition_task_2_times = {
        'Environment 1': defaultdict(list),
        'Environment 2': defaultdict(list),
    }

    # Walk through the directory structure
    for condition in condition_task_2_times.keys():
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
                                if 'task_2' in data_dict:
                                    for trial in data_dict['task_2']:
                                        time = data_dict['task_2'][trial]['time']
                                        condition_task_2_times[condition][trial].append(time)
                        except json.JSONDecodeError as e:
                            print(f"Error reading {file_path}: {e}")
                        except KeyError as e:
                            print(f"Missing expected key in {file_path}: {e}")

    # Calculate the average time per trial for task_2 within each condition
    average_times_per_trial = {
        condition: {
            trial: sum(times) / len(times)
            for trial, times in trials.items() if len(times) > 0
        }
        for condition, trials in condition_task_2_times.items()
    }

    return average_times_per_trial

# Specify the path to the 'experiment' directory
base_path = 'C:/Users/Jeffs/Desktop/experiments'

# Get the average time per trial for task_2 for each condition
average_times_per_trial = get_average_time_per_trial_for_task_2(base_path)
print(average_times_per_trial)

# Create a figure and axis for the line plot
fig, ax = plt.subplots()

# Define the conditions
conditions = list(average_times_per_trial.keys())

# Plot lines for each condition
for condition in conditions:
    trials = sorted(average_times_per_trial[condition].keys())
    times = [average_times_per_trial[condition][trial] for trial in trials]
    ax.plot(trials, times, label=condition)

# Add labels and title
ax.set_xlabel('Trials')
ax.set_ylabel('Average Time')
ax.set_title('Average Time per Trial for Task 2 in Each Condition')
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
