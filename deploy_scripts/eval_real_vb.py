
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
from multiprocessing.managers import SharedMemoryManager
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
from real_world.umi_env import UmiEnv
from real_world.real_inference_util import (get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
import json
from typing import Dict, Any

OmegaConf.register_new_resolver("eval", eval, replace=True)
@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--frequency', '-f', default=5, type=float, help="Control frequency in Hz.")
@click.option('--deploy_with_tactile', default=False, help='Whether to use tactile sensor when deploy.')
@click.option('--elgato_dev_path',  default=None, help='Path to elgato capture card, such as /dev/video0.')
@click.option('--tactile_camera_left_path', default='/dev/video2', help='Path to tactile left sensor, such as /dev/video2.')
@click.option('--tactile_camera_right_path', default='/dev/video4', help='Path to tactile right sensor, such as /dev/video4.')
@click.option('--gripper_dev_path', default='/dev/ttyUSB1', help="Gripper path, such as /dev/ttyUSB0.")
@click.option('--tactile_resolution', default=(640,480), type=tuple, help="Tactile resolution.")
@click.option('--vis_obs_img', default=True, help='Whether to visualize observation images.')
def main(input, frequency, deploy_with_tactile,
    elgato_dev_path, tactile_camera_left_path, tactile_camera_right_path, gripper_dev_path, tactile_resolution, vis_obs_img):

    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cuda', pickle_module=dill)
    cfg = payload['cfg']
    # todo:  need to modify, forgot the path of action's horizon in cfg
    # steps_per_inference = cfg.task.shape_meta.action.robot0_eef_pos.horizon
    steps_per_inference = 3
    print("steps_per_inference:", steps_per_inference)

    # setup experiment
    dt = 1/frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)

    with SharedMemoryManager() as shm_manager:
        with UmiEnv(
                deploy_with_tactile=deploy_with_tactile,
                elgato_dev_path=elgato_dev_path,
                tactile_camera_left_path=tactile_camera_left_path,
                tactile_camera_right_path=tactile_camera_right_path,
                gripper_dev_path=gripper_dev_path,
                tactile_resolution=tactile_resolution,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                # latency
                camera_obs_latency=0.17,
                robot_obs_latency=0.0001,
                gripper_obs_latency=0.01,
                robot_action_latency=0.18,
                gripper_action_latency=0.2,
                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

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
            pose = np.concatenate([
                obs[f'robot0_eef_pos'],
                obs[f'robot0_eef_rot_axis_angle']
            ], axis=-1)[-1]
            episode_start_pose.append(pose)
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep, episode_start_pose = episode_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 10
                action = get_real_umi_action(action, obs, action_pose_repr)
                assert action.shape[-1] == 7
                del result

            print('Ready!')
            while True:
                try:
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    # wait for 1/30 sec to get the closest frame (reduces overall latency)
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    while True:
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep, episode_start_pose = episode_start_pose)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            print('Inference latency:', time.time() - s)

                        this_target_poses = action
                        action_timestamps = (np.arange(len(action), dtype=np.float64)
                            ) * dt + obs_timestamps[-1]
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
                        print(this_target_poses)
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
                        if vis_obs_img:  # n 224 224 3
                            num_img = obs['camera0_rgb'].shape[0] 
                            for i in range(num_img):
                                gopro_img = obs['camera0_rgb'][i]
                                cv2.imshow('gopro_img', gopro_img)
                                if deploy_with_tactile:
                                    tac_left_img = obs['camera0_left_tactile'][i]
                                    tac_right_img = obs['camera0_right_tactile'][i]
                                    combined_img = np.hstack([gopro_img, tac_left_img, tac_right_img])
                                    cv2.imshow('combined_img', combined_img)
                                cv2.waitKey(10)

                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference                     

                except KeyboardInterrupt:
                    print("Interrupted!")
                
                print("Stopped.")


if __name__ == '__main__':
    main()