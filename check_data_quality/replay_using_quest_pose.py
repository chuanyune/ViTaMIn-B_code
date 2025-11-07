import sys
import os
import time
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
print(ROOT_DIR)
from real_world.rokae.rokae_interface import RokaeInterface
from real_world.pgi.pgi_interface import PGIInterface
import cv2
from diffusion_policy.common.replay_buffer import ReplayBuffer
import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

tx_quest_2_ee = np.load('./check_data_quality/tx_quest_2_ee.npy')

def get_transformation_matrix(position, rotation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position

    return transformation_matrix



def get_robot_pose_list(ee_pose_list: list, width_list: list):

    ee_pose_list = [ee_pose @ tx_quest_2_ee for ee_pose in ee_pose_list]
    ee_pose_0 = ee_pose_list[0]  # First pose
    base_2_ee_0 = np.linalg.inv(ee_pose_0)
    relative_pose_list = [base_2_ee_0 @ ee_pose for ee_pose in ee_pose_list]
    return relative_pose_list, width_list


def exec_arm(relative_pose_list_a, width_list_a):

    robot_a = RokaeInterface(arm_name='A')
    calibration_path = "./assets/cali_width_result/width_calibration.json" 
    pgi_a = PGIInterface(
        serial_name="/dev/ttyUSB0", 
        timeout=1, 
        calibration_file_path=calibration_path
    )

    curr_pose_a = robot_a.get_obs_replay

    next_pose_list_a = [curr_pose_a @ relative_pose for relative_pose in relative_pose_list_a]
    step_idx = 0
    for (pose_a, width_a) in zip(next_pose_list_a, width_list_a,):
        robot_a.execute_replay(pose_a)
        pgi_a.set_pos(width_a)
        print(f"step {step_idx}")
        step_idx += 1
        time.sleep(0.1)


if __name__ == "__main__":


    input = './single_green_bg_red_table_73.zarr.zip'
    with zarr.ZipStore(input, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, store=zarr.MemoryStore())
    # 4 70 60 49 30
    replay_episode = 30
    episode_slice = replay_buffer.get_episode_slice(replay_episode)
    start_idx = episode_slice.start
    stop_idx =  episode_slice.stop
    flag = 0
    traj_list = list()
    width_list = list()
    while True:
        pos = replay_buffer['robot0_eef_pos'][start_idx]
        rot = replay_buffer['robot0_eef_rot_axis_angle'][start_idx]
        traj = get_transformation_matrix(pos, rot)
        width = replay_buffer['robot0_gripper_width'][start_idx]
        traj_list.append(traj)
        width_list.append(width)
        if start_idx == stop_idx:
            break
        start_idx += 1

    print("ready to exec!!!!!!!!!!")
    relative_pose_list_a, width_list_a = get_robot_pose_list(traj_list, width_list)

    exec_arm(
        relative_pose_list_a, width_list_a,
    )