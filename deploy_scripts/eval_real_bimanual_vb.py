import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import os
import time
from multiprocessing.managers import SharedMemoryManager
from datetime import datetime
from pathlib import Path

import click
import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from utils.precise_sleep import precise_wait
from real_world.bimanual_umi_env import BimanualUmiEnv
from real_world.real_inference_util import (get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def save_observation_data(obs, save_dir, step_idx, deploy_with_tactile_img, deploy_with_tactile_pc):
    step_str = f"step_{step_idx:06d}"
    
    # Save visual images - directly in visual_images folder
    visual_dir = save_dir / "visual_images"
    
    for key in obs.keys():
        if key.endswith('_rgb'):
            # Visual RGB images - save the latest frame
            img_data = obs[key][-1]  # Get the latest frame
            if img_data.dtype == np.float32:
                img_data = (img_data * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV saving
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            save_path = visual_dir / f"{key}_{step_str}.png"
            cv2.imwrite(str(save_path), img_bgr)
    
    # Save tactile images if enabled - directly in tactile_images folder
    if deploy_with_tactile_img:
        tactile_img_dir = save_dir / "tactile_images"
        
        for key in obs.keys():
            if key.endswith('_tactile') and not key.endswith('_pc'):
                # Tactile images - save the latest frame
                img_data = obs[key][-1]  # Get the latest frame
                if img_data.dtype == np.float32:
                    img_data = (img_data * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV saving
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                save_path = tactile_img_dir / f"{key}_{step_str}.png"
                cv2.imwrite(str(save_path), img_bgr)
    
    # Save tactile point clouds if enabled - directly in tactile_point_clouds folder
    if deploy_with_tactile_pc:
        tactile_pc_dir = save_dir / "tactile_point_clouds"
        
        for key in obs.keys():
            if key.endswith('_tactile_points'):
                # Tactile point clouds - fixed array format
                pc_array = obs[key]  # Shape: (T, num_points, 3)
                # Get the latest frame
                pc_data = pc_array[-1]  # Shape: (num_points, 3)
                save_path = tactile_pc_dir / f"{key}_{step_str}.npy"
                np.save(str(save_path), pc_data)
    
    # Save robot state information - directly in visual_images folder
    robot_info = {}
    for key in obs.keys():
        if key.startswith('robot') and ('eef_pos' in key or 'eef_rot' in key or 'gripper_width' in key):
            robot_info[key] = obs[key][-1]  # Get the latest state
    
    if robot_info:
        robot_info_path = visual_dir / f"robot_state_{step_str}.npy"
        np.save(str(robot_info_path), robot_info)
    
    print(f"Saved observation data for step {step_idx} to {save_dir.name}")

@click.command()

@click.option('--input', '-i', default=None, help='Path to checkpoint')

@click.option('--deploy_with_tactile_img', '-dti', default=False, help='Deploy with tactile image sensor')
@click.option('--deploy_with_tactile_pc', '-dtpc', default=False, help='Deploy with tactile point cloud sensor')
@click.option('--fps_num_points', '-fnp', default=256, type=int, help='Number of points for FPS sampling in tactile point cloud')
@click.option('--save_obs', '-so', default=True, help='Save observation data for verification (saves every step)')
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--robots_name', '-rn', default=['A', 'B'], type=list, help="-")

@click.option('--gripper_ser_path', default=['/dev/ttyUSB1', '/dev/ttyUSB0'], type=list, help="-")

@click.option('--left_robot_cam_path', default=['/dev/video6', '/dev/video2', '/dev/video4'], type=list, help="-")
@click.option('--right_robot_cam_path', default=['/dev/video0', '/dev/video10', '/dev/video8'], type=list, help="-")

@click.option('--tactile_raw_data_resolution', default=(640, 480), type=tuple)

def main(input,
    deploy_with_tactile_img,
    deploy_with_tactile_pc,
    fps_num_points,
    save_obs,
    steps_per_inference,
    frequency, robots_name, gripper_ser_path, left_robot_cam_path, right_robot_cam_path, tactile_raw_data_resolution):

    # tx_right_left = np.array([
    #     [ 1,  0,  0, 25* 25 /1000],
    #     [  0, 1,  0,  0],
    #     [  0,  0,  1, 0],
    #     [  0,  0,  0, 1]])
    tx_right_left = np.array([
    [ 1,  0,  0, 36* 25 /1000],
    [  0, 1,  0,  0],
    [  0,  0,  1, 0],
    [  0,  0,  0, 1]])
    tx_robot1_robot0 = tx_right_left

    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    # print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    dt = 1/frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)

    print("steps_per_inference:", steps_per_inference)
    print(f"deploy_with_tactile_img: {deploy_with_tactile_img}")
    print(f"deploy_with_tactile_pc: {deploy_with_tactile_pc}")
    if deploy_with_tactile_pc:
        print(f"fps_num_points: {fps_num_points}")
    
    # Setup observation saving
    save_dir = None
    save_step_count = 0
    if save_obs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save in the ViTaMIn-B/observation_data directory
        project_root = Path(__file__).parent.parent  # Go up from deploy_scripts to ViTaMIn-B
        obs_data_dir = project_root / "observation_data"
        obs_data_dir.mkdir(exist_ok=True)  # Ensure the observation_data directory exists
        save_dir = obs_data_dir / f"saved_observations_{timestamp}"
        save_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different data types
        if deploy_with_tactile_img:
            (save_dir / "tactile_images").mkdir(exist_ok=True)
        if deploy_with_tactile_pc:
            (save_dir / "tactile_point_clouds").mkdir(exist_ok=True)
        (save_dir / "visual_images").mkdir(exist_ok=True)
        
        print(f"save_obs: {save_obs} (saves every step)")
        print(f"Observations will be saved to: {save_dir}")


    with SharedMemoryManager() as shm_manager:
        with BimanualUmiEnv(
                robots_name=robots_name,
                deploy_with_tactile_img=deploy_with_tactile_img,
                deploy_with_tactile_pc=deploy_with_tactile_pc,
                fps_num_points=fps_num_points,
                gripper_ser_path=gripper_ser_path,
                left_robot_cam_path=left_robot_cam_path,
                right_robot_cam_path=right_robot_cam_path,

                tactile_raw_data_resolution=tactile_raw_data_resolution,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                # latency
                camera_obs_latency=0.14,  # Visual camera latency
                tactile_obs_latency=0.08,  # Tactile camera latency
                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            
            # print("Waiting for camera")
            # time.sleep(3.0)

            # creating model
            # have to be done after fork to prevent 
            # duplicating CUDA context with ffmpeg nvenc
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)

            device = torch.device('cuda')
            policy.eval().to(device)

            print("Warming up policy inference")
            obs = env.get_obs()
            episode_start_pose = list()
            for robot_id in range(len(robots_name)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose)
                # Convert to torch tensors
                obs_dict = {}
                for key, value in obs_dict_np.items():
                    if isinstance(value, np.ndarray):
                        # Handle numpy.object_ arrays
                        if value.dtype == np.object_:
                            print(f"Warning: {key} has object dtype (shape: {value.shape}), attempting to convert...")
                            # Try to convert object array to numeric array
                            try:
                                # Stack all elements into a single array
                                stacked_list = []
                                for item in value.flat:
                                    if isinstance(item, np.ndarray):
                                        stacked_list.append(item)
                                    else:
                                        print(f"Error: {key} contains non-array element: {type(item)}")
                                        raise ValueError(f"Cannot convert {key} with non-array elements")
                                
                                if len(stacked_list) > 0:
                                    value = np.stack(stacked_list).astype(np.float32)
                                    print(f"Successfully converted {key} to shape: {value.shape}, dtype: {value.dtype}")
                                else:
                                    print(f"Error: {key} has no valid data to stack")
                                    continue
                            except Exception as e:
                                print(f"Failed to convert {key}: {e}")
                                import traceback
                                traceback.print_exc()
                                continue
                        
                        # Ensure array is contiguous and in a supported dtype
                        if not value.flags['C_CONTIGUOUS']:
                            value = np.ascontiguousarray(value)
                        
                        # Convert all numpy arrays to torch tensors and add batch dimension
                        obs_dict[key] = torch.from_numpy(value).unsqueeze(0).to(device)
                    else:
                        obs_dict[key] = value
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 10 * len(robots_name)
                action = get_real_umi_action(action, obs, action_pose_repr)
                assert action.shape[-1] == 7 * len(robots_name)
                del result

            print('Ready!')
            while True:
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay

                    # get current pose
                    obs = env.get_obs()

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')
                        
                        # Save observation data if enabled - save every step
                        if save_obs:
                            try:
                                save_observation_data(obs, save_dir, iter_idx, deploy_with_tactile_img, deploy_with_tactile_pc)
                                save_step_count += 1
                            except Exception as e:
                                print(f"Error saving observation data: {e}")

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                tx_robot1_robot0=tx_robot1_robot0,
                                episode_start_pose=episode_start_pose)
                            
                            # Convert to torch tensors
                            obs_dict = {}
                            for key, value in obs_dict_np.items():
                                if isinstance(value, np.ndarray):
                                    # Handle numpy.object_ arrays
                                    if value.dtype == np.object_:
                                        print(f"Warning: {key} has object dtype (shape: {value.shape}), attempting to convert...")
                                        # Try to convert object array to numeric array
                                        try:
                                            # Stack all elements into a single array
                                            stacked_list = []
                                            for item in value.flat:
                                                if isinstance(item, np.ndarray):
                                                    stacked_list.append(item)
                                                else:
                                                    print(f"Error: {key} contains non-array element: {type(item)}")
                                                    raise ValueError(f"Cannot convert {key} with non-array elements")
                                            
                                            if len(stacked_list) > 0:
                                                value = np.stack(stacked_list).astype(np.float32)
                                                print(f"Successfully converted {key} to shape: {value.shape}, dtype: {value.dtype}")
                                            else:
                                                print(f"Error: {key} has no valid data to stack")
                                                continue
                                        except Exception as e:
                                            print(f"Failed to convert {key}: {e}")
                                            import traceback
                                            traceback.print_exc()
                                            continue
                                    
                                    # Ensure array is contiguous and in a supported dtype
                                    if not value.flags['C_CONTIGUOUS']:
                                        value = np.ascontiguousarray(value)
                                    
                                    # Convert all numpy arrays to torch tensors and add batch dimension
                                    obs_dict[key] = torch.from_numpy(value).unsqueeze(0).to(device)
                                else:
                                    obs_dict[key] = value
                            
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            print('Inference latency:', time.time() - s)
                        
                        # convert policy action to env actions
                        this_target_poses = action
                        assert this_target_poses.shape[1] == len(robots_name) * 7


                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64)
                            ) * dt + obs_timestamps[-1]
                        print(dt)
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # execute actions

                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
                        
                        print(f"Submitted {len(this_target_poses)} steps of actions.")
 
                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    if save_obs and save_step_count > 0:
                        print(f"Saved {save_step_count} observation steps to {save_dir}")
                    break
                
                print("Stopped.")
                if save_obs and save_step_count > 0:
                    print(f"Total saved observation steps: {save_step_count}")



# %%
if __name__ == '__main__':
    main()