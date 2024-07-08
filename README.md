# Bachelor-Thesis
Pybullet and Panda Arm Robot VR

Description:

This project opens a simulation in which two Panda robot arms are loaded for bimanual tasks to be performed in the simulation. Two conditions can be loaded; 1. Teleoperation (robot mirrors movements) 2. Kinesthetic teaching (use grippers to manually move robots). The task that can be performed includes lifting up a tray with some objects in it. This simulation can be opened in the oculus quest 2 and oculus quest 3.

Installation: 

This project requires two libraries to be installed:

Install the project with the files edited:
download files from this GitHub repository

Original files can be found here:
Pybullet: https://github.com/bulletphysics/bullet3.git
Panda-gym: https://github.com/qgallouedec/panda-gym.git

To open the environment in VR:

Once all the installation is done follow these steps to open the environment in VR:
Open the oculus app and navigate to my devices and make sure the VR headset is connected.
In the headset enable Quest Link.
Open steam VR and make sure everything is connected.
In visual studio right click on  “App_PhysicsServer_SharedMemory.vcxproj” navigate to “debug” then “start new instance”
Run your code



Usage

Run the code in VR:
env = gym.make('PandaReach-v3', render_mode="vr")

Run the code in GUI (without VR):
env = gym.make('PandaReach-v3', render_mode="human")

To use a different environment/ task (change the name of the environment):
	env = gym.make('PandaPush-v3', render_mode="human")

Run VR_test.py to open the teleoperation environment and vrEvent.py for kinesthetic teaching.

When the environment loads, on the right handler, press the trigger button on the back of the handler and for the left handler press on the joystick, it is important to press them in these orders.








 
