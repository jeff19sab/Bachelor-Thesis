from typing import Any, Dict
import threading
import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class Reach(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,

    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.3, width=1.5, height=0.4, x_offset=-0.3, y_offset= 0.0)
        self.sim.create_table(length=0.5, width=1.5, height=0.4, x_offset=1.0, y_offset= 0.0)
        self.sim.physics_client.setVRCameraState([1.5,0,-1.0])
        self.sim.physics_client.loadURDF("C:/Users/Jeffs/Desktop/bullet3-master/bullet3-master/data/tray/traybox.urdf", [0, 0, 1], globalScaling=0.5)

        self.sim.physics_client.loadURDF("C:/Users/Jeffs/Desktop/bullet3-master/bullet3-master/data/tray/traybox.urdf",[1, 0, 1], globalScaling=0.5)
        self.sim.create_box(
            body_name="target3",
            half_extents=np.ones(3) * 0.04 / 2,
            mass=1.0,
            ghost=False,
            position=np.array([0,0,2]),
            rgba_color=np.array([1.1, 0.9, 0.1, 1.0]),
            lateral_friction=4.0
        )
        self.sim.create_box(
            body_name="target4",
            half_extents=np.ones(3) * 0.04 / 2,
            mass=1.0,
            ghost=False,
            position=np.array([1, 0, 2]),
            rgba_color=np.array([1.1, 0.9, 0.1, 1.0]),
            lateral_friction=4.0
        )
        self.pr2_gripper = self.sim.physics_client.loadURDF("pr2_gripper.urdf")
        self.pr2_gripper2 = self.sim.physics_client.loadURDF("pr2_gripper.urdf")
        self.sim.create_box(
            body_name="target1",
            half_extents=np.ones(3) * 0.04 / 2,
            mass=0.0,
            ghost=True,
            position=np.array([1.03843895, 0.19999209, 0.19739941]),
            rgba_color=np.array([1.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_box(
            body_name="target_2",
            half_extents=np.ones(3) * 0.04 / 2,
            mass=0.0,
            ghost=True,
            position=np.array([ 1.03843835, -0.20001539,  0.19739879])
,
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        jointPositions = [0.550569, 0.000000, 0.549657, 0.000000]
        jointPositions2 = [0.550569, 0.000000, 0.549657, 0.000000]

        # Resetting the grippers joint in the simulation
        for jointIndex in range(self.sim.physics_client.getNumJoints(self.pr2_gripper)):
            self.sim.physics_client.resetJointState(self.pr2_gripper, jointIndex, jointPositions[jointIndex])
            self.sim.physics_client.setJointMotorControl2(self.pr2_gripper, jointIndex,
                                                          self.sim.physics_client.POSITION_CONTROL,
                                                          targetPosition=0,
                                                          force=0)

        for jointIndex2 in range(self.sim.physics_client.getNumJoints(self.pr2_gripper2)):
             self.sim.physics_client.resetJointState(self.pr2_gripper2, jointIndex2, jointPositions2[jointIndex])
             self.sim.physics_client.setJointMotorControl2(self.pr2_gripper2, jointIndex2,
                                                           self.sim.physics_client.POSITION_CONTROL,
                                                           targetPosition=0,
                                                           force=0)
        self.pr2_cid = self.sim.physics_client.createConstraint(self.pr2_gripper, -1, -1, -1, self.sim.physics_client.JOINT_FIXED, [0, 0, 0.1], [0.2, 0, 0],
                                     [-0.65, -0.9500006, 0.2])
        self.pr2_cid2 = self.sim.physics_client.createConstraint(self.pr2_gripper,
                                      0,
                                      self.pr2_gripper,
                                      2,
                                      jointType=self.sim.physics_client.JOINT_GEAR,
                                      jointAxis=[0, 1, 0],
                                      parentFramePosition=[0, 0, 0],
                                      childFramePosition=[0, 0, 0])

        # Fixed joints is making them stick together
        self.pr2_cid3 = self.sim.physics_client.createConstraint(self.pr2_gripper2, -1, -1, -1,
                                                                 self.sim.physics_client.JOINT_FIXED, [0, 0, 0.1],
                                                                 [0.2, 0, 0],
                                                                 [0.500000, -0.900006, 0.2])
        self.pr2_cid4 = self.sim.physics_client.createConstraint(self.pr2_gripper2,
                                                                 0,
                                                                 self.pr2_gripper2,
                                                                 2,
                                                                 jointType=self.sim.physics_client.JOINT_GEAR,
                                                                 jointAxis=[0, 1, 0],
                                                                 parentFramePosition=[0, 0, 0],
                                                                 childFramePosition=[0, 0, 0])
        self.sim.physics_client.changeConstraint(self.pr2_cid2, gearRatio=1, erp=0.5, relativePositionTarget=0.5, maxForce=300)
        self.sim.physics_client.changeConstraint(self.pr2_cid4, gearRatio=1, erp=0.5, relativePositionTarget=0.5, maxForce=300)

        self.sim.physics_client.configureDebugVisualizer(self.sim.physics_client.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        self.sim.physics_client.configureDebugVisualizer(self.sim.physics_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self.sim.physics_client.configureDebugVisualizer(self.sim.physics_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self.sim.physics_client.configureDebugVisualizer(self.sim.physics_client.COV_ENABLE_GUI, 0)
        self.sim.physics_client.configureDebugVisualizer(self.sim.physics_client.COV_ENABLE_MOUSE_PICKING, 0)
        self.sim.physics_client.configureDebugVisualizer(self.sim.physics_client.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        self.sim.physics_client.configureDebugVisualizer(self.sim.physics_client.COV_ENABLE_VR_PICKING, 0)
        self.sim.physics_client.configureDebugVisualizer(self.sim.physics_client.COV_ENABLE_VR_RENDER_CONTROLLERS, 1)
        self.sim.physics_client.configureDebugVisualizer(self.sim.physics_client.COV_ENABLE_RENDERING, 1)

        CONTROLLER_ID = 0
        CONTROLLER_ID_2 = 0
        POSITION = 1
        ORIENTATION = 2
        ANALOG = 3
        BUTTONS = 6

        self.controllerId = -1

        print("waiting for VR controller trigger")
        while self.controllerId < 0:
            events = self.sim.physics_client.getVREvents()
            for e in (events):
                if e[BUTTONS][33] == self.sim.physics_client.VR_BUTTON_IS_DOWN:
                    self.controllerId = e[CONTROLLER_ID]
        print("Using controllerId=" + str(self.controllerId))

        self.controllerId2 = -1

        print("waiting for second VR controller trigger")
        while self.controllerId2 < 0:
            events = self.sim.physics_client.getVREvents()
            for e in (events):
                if e[BUTTONS][32] == self.sim.physics_client.VR_BUTTON_IS_DOWN:
                    self.controllerId2 = e[CONTROLLER_ID_2]
        print("Using second controllerId=" + str(self.controllerId2))

        def gripper_control_thread(self):

            while True:
                # keep the gripper centered/symmetric
                b = self.sim.physics_client.getJointState(self.pr2_gripper, 2)[0]
                self.sim.physics_client.setJointMotorControl2(self.pr2_gripper, 0,
                                                              self.sim.physics_client.POSITION_CONTROL,
                                                              targetPosition=b, force=3000)
                b_2 = self.sim.physics_client.getJointState(self.pr2_gripper2, 2)[0]
                self.sim.physics_client.setJointMotorControl2(self.pr2_gripper2, 0,
                                                              self.sim.physics_client.POSITION_CONTROL,
                                                              targetPosition=b_2, force=3000)

                events = self.sim.physics_client.getVREvents()
                for e in events:
                    if e[CONTROLLER_ID] == self.controllerId:
                        # sync the vr pr2 gripper with the vr controller position
                        self.sim.physics_client.changeConstraint(self.pr2_cid, e[POSITION], e[ORIENTATION],
                                                                 maxForce=5000)
                        relPosTarget = 1 - e[ANALOG]
                        # open/close the gripper, based on analogue
                        self.sim.physics_client.changeConstraint(self.pr2_cid2,
                                                                 gearRatio=1,
                                                                 erp=1,
                                                                 relativePositionTarget=relPosTarget,
                                                                 maxForce=3000)
                    if e[CONTROLLER_ID_2] == self.controllerId2:
                        # sync the vr pr2 gripper with the vr controller position
                        self.sim.physics_client.changeConstraint(self.pr2_cid3, e[POSITION], e[ORIENTATION],
                                                                 maxForce=5000)
                        relPosTarget = 1 - e[ANALOG]
                        # open/close the gripper, based on analogue
                        self.sim.physics_client.changeConstraint(self.pr2_cid4,
                                                                 gearRatio=1,
                                                                 erp=1,
                                                                 relativePositionTarget=relPosTarget,
                                                                 maxForce=3000)


        # Create and start the gripper control thread
        gripper_thread = threading.Thread(target=gripper_control_thread, args=(self,))
        gripper_thread.daemon = True  # Set as daemon thread to automatically exit when the main program ends
        gripper_thread.start()

    # Function to get the finger width
    def get_gripper_finger_width(self, robot_id):
        joint_states = self.sim.physics_client.getJointStates(robot_id, [0,2])
        finger_positions = [state[0] for state in joint_states]
        finger_width = sum(finger_positions)
        if finger_width <= 0.5:
            return -1
        else:
            return 1

    def get_robot_gripper(self):
        gripper_width = self.get_gripper_finger_width(self.pr2_gripper)
        gripper_width_2 = self.get_gripper_finger_width(self.pr2_gripper2)

        return gripper_width, gripper_width_2

    def get_handler_position(self):

        gripper_position = self.sim.physics_client.getBasePositionAndOrientation(self.pr2_gripper)[0]
        gripper_position_2 = self.sim.physics_client.getBasePositionAndOrientation(self.pr2_gripper2)[0]
        appended_gripper_pos = np.array([gripper_position[0],
                                         gripper_position[1],
                                         gripper_position[2],
                                         1.0])
        appended_gripper_pos_2 = np.array([gripper_position_2[0],
                                           gripper_position_2[1],
                                           gripper_position_2[2],
                                         1.0])


        return appended_gripper_pos, appended_gripper_pos_2


    def get_robot_position(self):
        human_pos_1, human_pos_2 = self.get_handler_position()
        transformation_matrix = np.array([[1, 0, 0, -0.3],
                                          [0, 1, 0, -0.95],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]])

        # Calculate the inverse of the transformation matrix
        inverse_transformation_matrix = [[1, 0, 0, 0.3],
                                     [0, 1, 0, 0.95],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]]
        if human_pos_1 is not None:
            final_1 = np.dot(inverse_transformation_matrix, human_pos_1)
            final_2 = np.dot(inverse_transformation_matrix, human_pos_2)
            gripper_width = self.get_gripper_finger_width(self.pr2_gripper)
            gripper_width_2 = self.get_gripper_finger_width(self.pr2_gripper2)
            final = np.array([final_1[0],
                              final_1[1],
                              final_1[2],
                              gripper_width,
                              final_2[0],
                              final_2[1],
                              final_2[2],
                              gripper_width_2
            ])
            return final

    def distance(self):
        target_position_1 = np.array([1.03843895, 0.19999209, 0.19739941]) #green cube
        target_position_2 = np.array([ 1.03843835, -0.20001539,  0.19739879]) #yellow cube

        handler1_position, handler2_position = self.get_handler_position()

        difference_handler1 = np.linalg.norm(handler1_position[:3] - target_position_1)
        difference_handler2 = np.linalg.norm(handler2_position[:3] - target_position_2)



        return difference_handler1, difference_handler2


    def get_obs(self) -> np.ndarray:
        return np.array([])

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def ee_pos(self):
        return self.get_ee_position()

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
