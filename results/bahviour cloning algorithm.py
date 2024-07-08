import os
import json

# Define the root directory you want to start from
root_dir = 'C:/Users/Jeffs/Desktop/experiments'
data_dict = {"task_1": [], "task_2": [], "task_3": []}

# Process only "condition_1"
condition = "condition 2"
condition_path = os.path.join(root_dir, condition)

if os.path.isdir(condition_path):
    for ID in os.listdir(condition_path):
        ID_path = os.path.join(condition_path, ID)
        if os.path.isdir(ID_path):
            for json_file in os.listdir(ID_path):
                if json_file.endswith('.json'):
                    json_file_path = os.path.join(ID_path, json_file)
                    # Assuming each JSON file contains data of interest
                    with open(json_file_path, 'r') as f:
                        data = json.load(f)
                        for task in data_dict.keys():
                            if task in data:
                                trials = data[task]
                                for trial_key, trial_data in trials.items():
                                    actions = trial_data.get('actions', [])
                                    data_dict[task].extend(actions)
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Assume data_dict contains data for multiple tasks
tasks = ["task_1", "task_2", "task_3"]

# Initialize lists to store training history for each task
history_dict = {}

# Loop through each task
for task in tasks:
    actions_data = np.array(data_dict[task])  # Convert to numpy array

    # Normalize input data
    scaler = StandardScaler()
    actions_data_scaled = scaler.fit_transform(actions_data)

    # Reshape actions_data for LSTM input
    actions_data_reshaped = actions_data_scaled.reshape(actions_data.shape[0], 1, actions_data.shape[1])

    # Define and compile your model
    model = Sequential([
        LSTM(64, input_shape=(1, actions_data.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(actions_data.shape[1])  # Output layer, adjust units to match input dimension
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Train your model
    history = model.fit(actions_data_reshaped, actions_data_scaled, epochs=50, batch_size=32, validation_split=0.3,
                        verbose=1)

    # Store the training history
    history_dict[task] = history

    # Print MAE and MSE for the task
    train_mse = history.history['loss'][-1]
    train_mae = history.history['mae'][-1]
    val_mse = history.history['val_loss'][-1]
    val_mae = history.history['val_mae'][-1]

    print(f"{task} Metrics:")
    print(f"  Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}")
    print(f"  Validation MSE: {val_mse:.4f}, Validation MAE: {val_mae:.4f}")
    print()

# Plotting the training and validation loss separately for all tasks
plt.figure(figsize=(12, 6))

# Plot training loss
for task, history in history_dict.items():
    plt.plot(history.history['loss'], label=f'{task} Training Loss')

plt.title('Training Loss for Different Tasks')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))

# Plot validation loss
for task, history in history_dict.items():
    plt.plot(history.history['val_loss'], label=f'{task} Validation Loss')

plt.title('Validation Loss for Different Tasks')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()





