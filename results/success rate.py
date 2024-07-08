import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

def get_success_rate_per_task(base_path):
    # Initialize dictionaries to store total successes and total trials for each task within each condition
    condition_task_success = {
        'Teleoperation': defaultdict(lambda: {'success_count': 0, 'trial_count': 0}),
        'Egocentric': defaultdict(lambda: {'success_count': 0, 'trial_count': 0}),
    }

    # Walk through the directory structure
    for condition in condition_task_success.keys():
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
                                            success = data_dict[task][trial]['success']
                                            condition_task_success[condition][task]['trial_count'] += 1
                                            if success:
                                                condition_task_success[condition][task]['success_count'] += 1
                        except json.JSONDecodeError as e:
                            print(f"Error reading {file_path}: {e}")
                        except KeyError as e:
                            print(f"Missing expected key in {file_path}: {e}")

    # Calculate the success rate for each task within each condition
    success_rates_per_condition = {
        condition: {
            task: (task_info['success_count'] / task_info['trial_count']) * 100
            for task, task_info in tasks.items() if task_info['trial_count'] > 0
        }
        for condition, tasks in condition_task_success.items()
    }

    return condition_task_success, success_rates_per_condition

# Specify the path to the 'experiment' directory
base_path = 'C:/Users/Jeffs/Desktop/experiments'

# Get the success counts and success rate per task for each condition
condition_task_success, success_rates = get_success_rate_per_task(base_path)

# Calculate p-values for each task
p_values = {}
for task in ['task_1', 'task_2', 'task_3']:
    successes = [condition_task_success[env][task]['success_count'] for env in ['Teleoperation', 'Egocentric']]
    trials = [condition_task_success[env][task]['trial_count'] for env in ['Teleoperation', 'Egocentric']]

    # Create a contingency table
    contingency_table = [
        [successes[0], trials[0] - successes[0]],  # Environment 1
        [successes[1], trials[1] - successes[1]]  # Environment 2
    ]

    # Perform chi-square test or Fisher's exact test based on sample size
    if min(trials) >= 5:  # Chi-square test is reliable when sample sizes are 5 or more
        chi2, p, _, _ = chi2_contingency(contingency_table)
    else:  # Fisher's exact test is used for small sample sizes
        _, p = fisher_exact(contingency_table)

    p_values[task] = p

# Calculate error bars (standard error) for each task and condition
error_bars = {}
for task in ['task_1', 'task_2', 'task_3']:
    errors = []
    for env in ['Teleoperation', 'Egocentric']:
        success_count = condition_task_success[env][task]['success_count']
        trial_count = condition_task_success[env][task]['trial_count']
        success_rate = success_count / trial_count
        standard_error = np.sqrt(success_rate * (1 - success_rate) / trial_count) * 100  # Error in percentage
        errors.append(standard_error)
    error_bars[task] = errors

# Print success counts, trial counts, and success rates for each task and condition
for condition in condition_task_success.keys():
    print(f"Condition: {condition}")
    for task in ['task_1', 'task_2', 'task_3']:
        successes = condition_task_success[condition][task]['success_count']
        trials = condition_task_success[condition][task]['trial_count']
        success_rate = success_rates[condition][task]

        print(f"Task: {task}")
        print(f"  Successes: {successes}")
        print(f"  Trials: {trials}")
        print(f"  Success Rate: {success_rate:.2f}%")
        print()

# Create a figure and axis for the success rate plot
fig, ax = plt.subplots()

# Define the tasks and conditions
tasks = ['task_1', 'task_2', 'task_3']
conditions = list(success_rates.keys())

# Define bar width and positions
bar_width = 0.35
index = np.arange(len(tasks))

# Plot bars for each condition
for i, condition in enumerate(conditions):
    rates = [success_rates[condition][task] for task in tasks]
    errors = [error_bars[task][i] for task in tasks]  # Use the calculated error bars for each task
    ax.bar(index + i * bar_width, rates, bar_width, yerr=errors, capsize=5, label=condition, alpha=0.7)

# Add labels and title
ax.set_xlabel('Tasks')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Success Rate per Task for Each Condition')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(tasks)
ax.legend()

# Annotate p-values on the plot
for i, task in enumerate(tasks):
    p_val = p_values.get(task, None)
    if p_val is not None:
        # Adjust text position and add background color
        ax.text(index[i] + bar_width / 2, max(success_rates['Teleoperation'][task], success_rates['Egocentric'][task]) + 7,
                f'p={p_val:.3f}', ha='left', fontsize=10, color='black', weight='bold')

# Show the plot
plt.tight_layout()
plt.show()
# Print p-values
print("p-values for each task:")
for task, p_value in p_values.items():
    print(f"{task}: p={p_value:.3f}")
