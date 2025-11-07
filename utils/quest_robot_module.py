import os
import time
import datetime
import socket
import numpy as np
from scipy.spatial.transform import Rotation

class QuestRobotModule:
    def __init__(self, vr_ip, local_ip, pose_cmd_port, ik_result_port=None):
        self.vr_ip = vr_ip
        self.local_ip = local_ip
        self.pose_cmd_port = pose_cmd_port
        # Quest should send WorldFrame as well as wrist pose via UDP
        self.wrist_listener_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.wrist_listener_s.bind(("", pose_cmd_port))
        self.wrist_listener_s.setblocking(1)
        self.wrist_listener_s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 0)
        # Initialize ik sender to Quest
        if ik_result_port is not None:
            self.ik_result_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.ik_result_dest = (vr_ip, ik_result_port)
        else:
            self.ik_result_s = None
    def compute_rel_transform(self, pose):
        """
        pose: np.ndarray shape (7,) [x, y, z, qx, qy, qz, qw] in unity frame
        """
        world_frame = self.world_frame.copy()
        world_frame[:3] = np.array([world_frame[0], world_frame[2], world_frame[1]])
        pose[:3] = np.array([pose[0], pose[2], pose[1]])

        Q = np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0.]])
        rot_base = Rotation.from_quat(world_frame[3:]).as_matrix()
        rot = Rotation.from_quat(pose[3:]).as_matrix()
        rel_rot = Rotation.from_matrix(Q @ (rot_base.T @ rot) @ Q.T) # Is order correct.
        rel_pos = Rotation.from_matrix(Q @ rot_base.T@ Q.T).apply(pose[:3] - world_frame[:3]) # Apply base rotation not relative rotation...
        return rel_pos, rel_rot.as_quat()
    
    def close(self):
        self.wrist_listener_s.close()
        if self.ik_result_s is not None:
            self.ik_result_s.close()

class QuestLeftArmGripperModule(QuestRobotModule):
    def __init__(self, vr_ip, local_ip, pose_cmd_port, ik_result_port, vis_sp=None):
        super().__init__(vr_ip, local_ip, pose_cmd_port, ik_result_port)
        self.vis_sp = vis_sp
        self.data_dir = None
        self.prev_data_dir = self.data_dir
        self.last_arm_q = None
        self.last_hand_q = None
        self.last_action = 1
        self.last_action_t = time.time()

    def receive(self):
        data, _ = self.wrist_listener_s.recvfrom(1024)
        data_string = data.decode()
        # print(data_string)
        # now = datetime.datetime.now()
        if data_string.startswith("WorldFrame"):
            print("\033[95m[SYSTEM] Received WorldFrame\033[0m")
            data_string = data_string[11:]
            data_string = data_string.split(",")
            data_list = [float(data) for data in data_string]
            world_frame = np.array(data_list)
            self.world_frame = world_frame
            return None, None, None
        elif data_string.startswith("Start"):
            return None, None, None
        elif data_string.startswith("Stop"):
            return None, None, None
        elif data_string.startswith("Remove"):
            return None, None, None
        elif data_string.find("YLHand") != -1:
            data_string_ = data_string[7:].split(",")
            data_list = [float(data) for data in data_string_]
            wrist_tf = np.array(data_list[:7])
            head_tf = np.array(data_list[7:14])
            timestamp = data_list[14]
            rel_wrist_pos, rel_wrist_rot = self.compute_rel_transform(wrist_tf)
            rel_head_pos, rel_head_rot = self.compute_rel_transform(head_tf)
            return (rel_wrist_pos, rel_wrist_rot), (rel_head_pos, rel_head_rot), timestamp
        elif data_string.find("NLHand") != -1:
            return None, None, None
        else:
            return None, None, None

class QuestBimanualModule(QuestRobotModule):
    def __init__(self, vr_ip, local_ip, pose_cmd_port, ik_result_port, vis_sp=None):
        super().__init__(vr_ip, local_ip, pose_cmd_port, ik_result_port)
        self.vis_sp = vis_sp
        self.data_dir = None
        self.prev_data_dir = self.data_dir
        self.last_arm_q = None
        self.last_hand_q = None
        self.last_action = 1
        self.last_action_t = time.time()

    def receive(self):
        data, _ = self.wrist_listener_s.recvfrom(1024)
        data_string = data.decode()
        #print(data_string)
        # now = datetime.datetime.now()
        if data_string.startswith("WorldFrame"):
            print("\033[95m[SYSTEM] Received WorldFrame\033[0m")
            data_string = data_string[11:]
            data_string = data_string.split(",")
            data_list = [float(data) for data in data_string]
            world_frame = np.array(data_list)
            self.world_frame = world_frame
            return None, None, None
        elif data_string.startswith("RobotFrame"):
            print("Robot frame received")
            return None, None, None
        elif data_string.startswith("Start"):
            return None, None, None
        elif data_string.startswith("Stop"):
            return None, None, None
        elif data_string.startswith("Remove"):
            return None, None, None
        elif data_string.find("YBHand") != -1:
            data_string_ = data_string[7:].split(",")
            data_list = [float(data) for data in data_string_]
            left_wrist_tf = np.array(data_list[:7])
            right_wrist_tf = np.array(data_list[7:14])
            head_tf = np.array(data_list[14:])
            rel_left_wrist_pos, rel_left_wrist_rot = self.compute_rel_transform(left_wrist_tf)
            rel_right_wrist_pos, rel_right_wrist_rot = self.compute_rel_transform(right_wrist_tf)
            rel_head_pos, rel_head_rot = self.compute_rel_transform(head_tf)
            return (rel_left_wrist_pos, rel_left_wrist_rot), (rel_right_wrist_pos, rel_right_wrist_rot), (rel_head_pos, rel_head_rot)
        else:
            return None, None, None
