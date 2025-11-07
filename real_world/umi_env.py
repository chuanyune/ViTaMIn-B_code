from typing import Tuple
import numpy as np
import time
import math
from multiprocessing.managers import SharedMemoryManager
from real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder
from diffusion_policy.common.cv2_util import get_image_transform
from utils.interpolation_util import get_interp1d, PoseInterpolator
from real_world.rokae.rokae_interpolation_controller import RokaeInterpolationController
from real_world.pgi.pgi_controller import PGIController
from utils.cv_util import draw_fisheye_mask

class UmiEnv:
    def __init__(self, 
                 deploy_with_tactile=False,
                 elgato_dev_path=None,
                 tactile_camera_left_path=None,
                 tactile_camera_right_path=None,
                 gripper_dev_path=None,
                 tactile_resolution: Tuple[int, int] =(640,480),
                 frequency=None,
                 obs_image_resolution=(224,224),
                 max_obs_buffer_size=60,
                 obs_float32=False,
                 # timing
                 align_camera_idx=0,
                 # this latency compensates receive_timestamp
                 # all in seconds
                 camera_obs_latency=0.125,
                 robot_obs_latency=0.0001,
                 gripper_obs_latency=0.01,
                 robot_action_latency=0.1,
                 gripper_action_latency=0.1,
                 # all in steps (relative to frequency)
                 camera_down_sample_steps=1,
                 robot_down_sample_steps=1,
                 gripper_down_sample_steps=1,
                 use_fisheye_mask=True,
                 fisheye_mask_radius=390,
                 fisheye_mask_center=None,
                 fisheye_mask_fill_color=(0, 0, 0),
                 # all in steps (relative to frequency)
                 camera_obs_horizon=2,
                 robot_obs_horizon=2,
                 gripper_obs_horizon=2,
                 # shared memory
                 shm_manager=None
                 ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        # Wait for all v4l cameras to be back online
        time.sleep(0.1)
        if deploy_with_tactile:
            v4l_paths = [elgato_dev_path, tactile_camera_left_path, tactile_camera_right_path]
        else:
            v4l_paths = [elgato_dev_path]

        # HACK: Separate video setting for each camera
        resolution = list()
        capture_fps = list()
        cap_buffer_size = list()
        video_recorder = list()
        transform = list()
        for idx in range(len(v4l_paths)):
            if idx == 0:
                res = (1920, 1080)
                fps = 30
                buf = 3
                bit_rate = 6000*1000
                def tf4k(data, input_res=res):
                    img = data['color']
                    if use_fisheye_mask:
                        img = draw_fisheye_mask(
                            img, 
                            radius=fisheye_mask_radius,
                            center=fisheye_mask_center,
                            fill_color=fisheye_mask_fill_color
                        )
                    
                    f = get_image_transform(
                        input_res=input_res,
                        output_res=obs_image_resolution, 
                        # obs output rgb
                        bgr_to_rgb=True)
                    img = f(img)
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf4k)
            else:
                res = tactile_resolution
                fps = 30
                buf = 1
                bit_rate = 3000*1000
                def tf(data, input_res=res):
                    img = data['color']
                    f = get_image_transform(
                        input_res=input_res,
                        output_res=obs_image_resolution, 
                        # obs output rgb
                        bgr_to_rgb=True)
                    img = np.ascontiguousarray(f(img))
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf)

            resolution.append(res)
            capture_fps.append(fps)
            cap_buffer_size.append(buf)
            video_recorder.append(VideoRecorder.create_hevc_nvenc(
                fps=fps,
                input_pix_fmt='bgr24',
                bit_rate=bit_rate
            ))

        camera = MultiUvcCamera(
            dev_video_paths=v4l_paths,
            shm_manager=shm_manager,
            resolution=resolution,
            capture_fps=capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            get_max_k=max_obs_buffer_size,
            receive_latency=camera_obs_latency,
            cap_buffer_size=cap_buffer_size,
            transform=transform,
            video_recorder=video_recorder,
            verbose=False
        )

        robot = RokaeInterpolationController(
            shm_manager=shm_manager,
            frequency=30, 
            verbose=True,
            arm_name="A"
        )

        gripper = PGIController(
            shm_manager=shm_manager,
            ser=gripper_dev_path,
            frequency=30,
            verbose=True,
            calibration_path='./assets/cali_width_result/width_calibration.json'
        )

        self.camera = camera
        self.robot = robot
        self.gripper = gripper
        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        # timing
        self.align_camera_idx = align_camera_idx
        self.camera_obs_latency = camera_obs_latency
        self.robot_obs_latency = robot_obs_latency
        self.gripper_obs_latency = gripper_obs_latency
        self.robot_action_latency = robot_action_latency
        self.gripper_action_latency = gripper_action_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        # temp memory buffers
        self.last_camera_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.start_time = None
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.camera.is_ready and self.robot.is_ready and self.gripper.is_ready
    
    def start(self, wait=True):
        self.camera.start(wait=False)
        self.gripper.start(wait=False)
        self.robot.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.robot.stop(wait=False)
        self.gripper.stop(wait=False)
        self.camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.camera.start_wait()
        self.gripper.start_wait()
        self.robot.start_wait()
    
    def stop_wait(self):
        self.robot.stop_wait()
        self.gripper.stop_wait()
        self.camera.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        """
        Timestamp alignment policy
        'current' time is the last timestamp of align_camera_idx
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time
        """

        "observation dict"
        assert self.is_ready

        # get data
        # 60 Hz, camera_calibrated_timestamp
        k = math.ceil(
            self.camera_obs_horizon * self.camera_down_sample_steps \
            * (60 / self.frequency))
        self.last_camera_data = self.camera.get(
            k=k, 
            out=self.last_camera_data)

        # 125/500 hz, robot_receive_timestamp
        last_robot_data = self.robot.get_all_state()
        # both have more than n_obs_steps data

        # 30 hz, gripper_receive_timestamp
        last_gripper_data = self.gripper.get_all_state()

        last_timestamp = self.last_camera_data[self.align_camera_idx]['timestamp'][-1]
        dt = 1 / self.frequency

        # align camera obs timestamps
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in camera_obs_timestamps:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                this_idxs.append(nn_idx)
            # remap key
            if camera_idx == 0:
                camera_obs['camera0_rgb'] = value['color'][...,:3][this_idxs]
            elif camera_idx == 1: # 
                camera_obs['camera0_left_tactile'] = value['color'][...,:3][this_idxs]
            elif camera_idx == 2:
                camera_obs['camera0_right_tactile'] = value['color'][...,:3][this_idxs]
            else:
                raise ValueError(f'Invalid camera index: {camera_idx}')

        # align robot obs
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        robot_pose_interpolator = PoseInterpolator(
            t=last_robot_data['robot_timestamp'], 
            x=last_robot_data['ee_pose'])
        robot_pose = robot_pose_interpolator(robot_obs_timestamps)
        robot_obs = {
            'robot0_eef_pos': robot_pose[...,:3],
            'robot0_eef_rot_axis_angle': robot_pose[...,3:]
        }

        # align gripper obs
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
        gripper_interpolator = get_interp1d(
            t=last_gripper_data['gripper_timestamp'],
            x=last_gripper_data['gripper_position'][...,None]
        )
        gripper_obs = {
            'robot0_gripper_width': gripper_interpolator(gripper_obs_timestamps)
        }

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data.update(gripper_obs)
        obs_data['timestamp'] = camera_obs_timestamps

        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        r_latency = self.robot_action_latency if compensate_latency else 0.0
        g_latency = self.gripper_action_latency if compensate_latency else 0.0

        # schedule waypoints
        for i in range(len(new_actions)):
            r_actions = new_actions[i,:6]
            g_actions = new_actions[i,6:]
            self.robot.schedule_waypoint(
                pose=r_actions,
                target_time=new_timestamps[i]-r_latency
            )
            self.gripper.schedule_waypoint(
                pos=g_actions,
                target_time=new_timestamps[i]-g_latency
            )

    def get_robot_state(self):
        return self.robot.get_state()