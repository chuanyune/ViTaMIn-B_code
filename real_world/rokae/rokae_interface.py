import rospy
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R
from real_world.rokae.xMate3_interface import Cartesian_control_interface
from utils.pose_util import pose_to_mat, mat_to_pose

import rospy

class RokaeInterface:
    def __init__(self, arm_name: str):
        if not rospy.core.is_initialized():
            rospy.init_node("umi_", anonymous=True)
        self.cci = Cartesian_control_interface([600 for i in range(6)], arm_name)
    
    def execute(self, xyz_rotvec: np.ndarray, Delta:bool):
        xyz_rotvec += self.get_obs * int(Delta)
        pose_mat = pose_to_mat(xyz_rotvec)
        self.cci.exec_arm(np.reshape(pose_mat, (16,)))

    def execute_replay(self, pose_mat):
        self.cci.exec_arm(np.reshape(pose_mat, (16,)))

    def stop(self, reason : str = None):
        if reason == None:
            rospy.signal_shutdown("Terminating node")
        else:
            rospy.signal_shutdown(reason)

    def check(self) -> bool:
        q_m = self.get_qm
        threshold = 1.5
        min_qm = np.array([-165, -115, -166, -120, -165, -120, -355]) + np.full((7,), threshold)
        max_qm = np.array([165, 115, 166, 120, 165, 120, 355]) - np.full((7,), threshold)
        return np.all((q_m >= min_qm) & (q_m <= max_qm))

    @property
    def get_obs(self) -> np.ndarray:
        current_mat = np.array(self.cci.arm_state.toolTobase_pos_m).reshape(4,4)
        return mat_to_pose(current_mat)

    @property
    def get_start_pose(self) -> np.ndarray:
        current_mat = np.array(self.cci.arm_state.toolTobase_pos_m).reshape(4,4)
        return mat_to_pose(current_mat)  # N * 6


    @property
    def get_obs_replay(self) -> np.ndarray:
        current_mat = np.array(self.cci.arm_state.toolTobase_pos_m).reshape(4,4)
        return current_mat

    @property
    def get_qm(self) -> np.ndarray:
        return np.degrees(self.cci.arm_state.q_m)

    @property
    def pose_current_mat(self) -> np.ndarray:
        return np.array(self.cci.arm_state.toolTobase_pos_m).reshape(4,4)