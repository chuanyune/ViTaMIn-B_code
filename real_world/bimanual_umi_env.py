from typing import Optional, List
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from real_world.rokae.rokae_interpolation_controller import RokaeInterpolationController
from real_world.pgi.pgi_controller import PGIController
from real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder
from diffusion_policy.common.cv2_util import get_image_transform
from utils.interpolation_util import get_interp1d, PoseInterpolator
from utils.cv_util import draw_fisheye_mask
import cv2
import time

class BimanualUmiEnv:
    def __init__(self, 
            # required params
            robots_name,
            deploy_with_tactile_img=False,
            deploy_with_tactile_pc=False,
            gripper_ser_path=None,
            left_robot_cam_path=None,
            right_robot_cam_path=None,
            tactile_raw_data_resolution=None,
            # tactile point cloud params
            fps_num_points=256,
            # env params
            frequency=20,
            # obs
            obs_image_resolution=(224,224),
            max_obs_buffer_size=60,
            obs_float32=False,
            # this latency compensates receive_timestamp
            # all in seconds
            camera_obs_latency=0.125,
            # separate latency for tactile cameras (optional)
            tactile_obs_latency=None,
            # all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            # all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            # fisheye mask params (consistent with training data)
            use_fisheye_mask=True,
            fisheye_mask_radius=390,
            fisheye_mask_center=None,
            fisheye_mask_fill_color=(0, 0, 0),
            # shared memory
            shm_manager=None
            ):

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        # Wait for all v4l cameras to be back online
        time.sleep(0.1)
        
        # Determine camera configuration based on tactile settings
        deploy_with_tactile = deploy_with_tactile_img or deploy_with_tactile_pc
        if deploy_with_tactile:
            v4l_paths = left_robot_cam_path + right_robot_cam_path
        else:
            v4l_paths = [left_robot_cam_path[0], right_robot_cam_path[0]]

        resolution = list()
        capture_fps = list()
        cap_buffer_size = list()
        video_recorder = list()
        transform = list()
        if deploy_with_tactile:
            for idx in range(len(v4l_paths)):
                if idx == 0 or idx == 3:
                    res = (1920, 1080)
                    fps = 30
                    buf = 3
                    bit_rate = 6000*1000
                    def tf4k(data, input_res=res, 
                            use_mask=use_fisheye_mask, 
                            mask_radius=fisheye_mask_radius,
                            mask_center=fisheye_mask_center,
                            mask_fill_color=fisheye_mask_fill_color,
                            camera_idx=idx):
                        # Apply fisheye mask processing (consistent with training data)
                        img = data['color']
                        
                        # Apply mask only to visual cameras (tactile cameras don't need mask)
                        if use_mask and (camera_idx == 0 or camera_idx == 3):  # visual cameras
                            # Apply mask before resize, consistent with training data processing order
                            img = draw_fisheye_mask(
                                img, 
                                radius=mask_radius,
                                center=mask_center,
                                fill_color=mask_fill_color
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
                    res = tactile_raw_data_resolution
                    fps = 30
                    buf = 1
                    bit_rate = 3000*1000
                    def tf(data, input_res=res, 
                          deploy_img=deploy_with_tactile_img, 
                          deploy_pc=deploy_with_tactile_pc):
                        img = data['color']
                        
                        # If point cloud mode only, keep original size and uint8 format for point cloud extraction
                        if deploy_pc and not deploy_img:
                            # Point cloud mode: use same input/output size transform, only color conversion, keep uint8 format
                            f = get_image_transform(
                                input_res=input_res,
                                output_res=input_res,  # input and output size are the same
                                # obs output rgb
                                bgr_to_rgb=True)
                            img = np.ascontiguousarray(f(img))
                            # Don't convert to float32, keep uint8 format
                            data['color'] = img
                        else:
                            # Image mode or hybrid mode: normal resize and float conversion
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
                video_recorder.append(VideoRecorder.create_hevc_nvenc(  # TODO: why use hevc
                    fps=fps,
                    input_pix_fmt='bgr24',
                    bit_rate=bit_rate
                ))
        else:
            for idx in range(len(v4l_paths)):
                res = (1920, 1080)
                fps = 30
                buf = 3
                bit_rate = 6000*1000
                def tf4k(data, input_res=res,
                        use_mask=use_fisheye_mask,
                        mask_radius=fisheye_mask_radius,
                        mask_center=fisheye_mask_center,
                        mask_fill_color=fisheye_mask_fill_color):
                    # Apply fisheye mask processing (consistent with training data)
                    img = data['color']
                    
                    # Apply mask to all visual cameras (non-tactile mode has only visual cameras)
                    if use_mask:
                        # Apply mask before resize, consistent with training data processing order
                        img = draw_fisheye_mask(
                            img, 
                            radius=mask_radius,
                            center=mask_center,
                            fill_color=mask_fill_color
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
            
                resolution.append(res)
                capture_fps.append(fps)
                cap_buffer_size.append(buf)
                video_recorder.append(VideoRecorder.create_hevc_nvenc(  # TODO: why use hevc
                    fps=fps,
                    input_pix_fmt='bgr24',
                    bit_rate=bit_rate
                ))

        # Configure different latencies for different camera types
        camera_latencies = []
        for idx in range(len(v4l_paths)):
            if deploy_with_tactile:
                # For bimanual setup: [visual_left, tactile_left_1, tactile_left_2, visual_right, tactile_right_1, tactile_right_2]
                if idx in [0, 3]:  # Visual cameras (camera indices 0 and 3)
                    camera_latencies.append(camera_obs_latency)
                else:  # Tactile cameras (camera indices 1, 2, 4, 5)
                    latency = tactile_obs_latency if tactile_obs_latency is not None else camera_obs_latency
                    camera_latencies.append(latency)
            else:
                # For non-tactile setup: all cameras use same latency
                camera_latencies.append(camera_obs_latency)
        
        # Configure tactile point cloud parameters for each camera
        enable_tactile_pc_list = []
        fps_num_points_list = []
        tactile_lower_bound_list = []
        
        for idx in range(len(v4l_paths)):
            if deploy_with_tactile and deploy_with_tactile_pc:
                # For bimanual setup: [visual_left, tactile_left_1, tactile_left_2, visual_right, tactile_right_1, tactile_right_2]
                if idx in [0, 3]:  # Visual cameras (camera indices 0 and 3)
                    enable_tactile_pc_list.append(False)
                    fps_num_points_list.append(256)  # Default value, not used
                    tactile_lower_bound_list.append(10)  # Default value, not used
                else:  # Tactile cameras (camera indices 1, 2, 4, 5)
                    enable_tactile_pc_list.append(True)
                    fps_num_points_list.append(fps_num_points)
                    # Use exact same lower_bound as non-parallel version for each camera
                    if idx == 1:  # Left robot tactile camera 1 (left_hand_left_tactile)
                        tactile_lower_bound_list.append(25)  # Consistent with non-parallel version
                    elif idx == 2:  # Left robot tactile camera 2 (left_hand_right_tactile)
                        tactile_lower_bound_list.append(25)  # Consistent with non-parallel version
                    elif idx == 4:  # Right robot tactile camera 1 (right_hand_left_tactile)
                        tactile_lower_bound_list.append(25)  # Key fix: consistent with non-parallel version
                    elif idx == 5:  # Right robot tactile camera 2 (right_hand_right_tactile)
                        tactile_lower_bound_list.append(25)  # Consistent with non-parallel version
                    else:
                        # Fallback for unexpected indices
                        tactile_lower_bound_list.append(10)
            else:
                # For non-tactile setup or tactile image only: disable point cloud computation
                enable_tactile_pc_list.append(False)
                fps_num_points_list.append(256)  # Default value, not used
                tactile_lower_bound_list.append(10)  # Default value, not used

        camera = MultiUvcCamera(
            dev_video_paths=v4l_paths,
            shm_manager=shm_manager,
            resolution=resolution,
            capture_fps=capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            get_max_k=max_obs_buffer_size,
            receive_latency=camera_latencies,
            cap_buffer_size=cap_buffer_size,
            transform=transform,
            video_recorder=video_recorder,
            # tactile point cloud params
            enable_tactile_pc=enable_tactile_pc_list,
            fps_num_points=fps_num_points_list,
            tactile_lower_bound=tactile_lower_bound_list,
            verbose=False
        )

        robots: List[RokaeInterpolationController] = list()
        grippers: List[PGIController] = list()
        for idx, (rn, gs) in enumerate(zip(robots_name, gripper_ser_path)):
            if idx == 0:
                this_robot = RokaeInterpolationController(
                    shm_manager=shm_manager,
                    frequency=30, 
                    verbose=True,
                    arm_name=rn)
            else:
                this_robot = RokaeInterpolationController(
                    shm_manager=shm_manager,
                    frequency=30, 
                    verbose=True,
                    arm_name=rn)

            this_gripper = PGIController(
                shm_manager=shm_manager,
                ser=gs,
                frequency=30,
                verbose=True,
                calibration_path="/home/drj/codehub/ViTaMIn-B/assets/cali_width_result/width_calibration.json")
            
            robots.append(this_robot)
            grippers.append(this_gripper)

        print(robots[0] is robots[1])
        self.camera = camera
        self.robots = robots
        self.robots_name = robots_name
        self.grippers = grippers
        self.gripper_ser_path = gripper_ser_path
        self.v4l_paths = v4l_paths
        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.deploy_with_tactile = deploy_with_tactile
        self.deploy_with_tactile_img = deploy_with_tactile_img
        self.deploy_with_tactile_pc = deploy_with_tactile_pc
        self.fps_num_points = fps_num_points
        
        # timing
        self.camera_obs_latency = camera_obs_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        
        # Store camera latencies for alignment reference
        self.camera_latencies = camera_latencies
        
        
        # temp memory buffers
        self.last_camera_data = None
        # recording buffers
        self.start_time = None
        self.last_time_step = 0
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        ready_flag = self.camera.is_ready
        for robot in self.robots:
            ready_flag = ready_flag and robot.is_ready
        for gripper in self.grippers:
            ready_flag = ready_flag and gripper.is_ready
        return ready_flag
    
    def start(self, wait=True):
        self.camera.start(wait=False)
        for robot in self.robots:
            robot.start(wait=False)
        for gripper in self.grippers:
            gripper.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        for robot in self.robots:
            robot.stop(wait=False)
        for gripper in self.grippers:
            gripper.stop(wait=False)
        self.camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.camera.start_wait()
        for robot in self.robots:
            robot.start_wait()
        for gripper in self.grippers:
            gripper.start_wait()
    
    def stop_wait(self):
        for robot in self.robots:
            robot.stop_wait()
        for gripper in self.grippers:
            gripper.stop_wait()
        self.camera.stop_wait()
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_camera_latency(self, camera_idx):
        """Get the latency compensation value for a specific camera"""
        return self.camera_latencies[camera_idx]
    

    # ========= async env API ===========
    def get_obs(self) -> dict:
        """
        Timestamp alignment policy
        We assume the cameras used for obs are always [0, k - 1], where k is the number of robots
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time
        """

        "observation dict"
        assert self.is_ready

        # get data
        # 60 Hz, camera_calibrated_timestamp (note: cameras capture at 30Hz, but using 60Hz for interpolation)
        k = math.ceil(
            self.camera_obs_horizon * self.camera_down_sample_steps \
            * (60 / self.frequency)) + 2 # here 2 is adjustable, typically 1 should be enough
        # print('==>k  ', k, self.camera_obs_horizon, self.camera_down_sample_steps, self.frequency)
        self.last_camera_data = self.camera.get(
            k=k, 
            out=self.last_camera_data)

        # both have more than n_obs_steps data
        last_robots_data = list()
        last_grippers_data = list()
        # 125/500 hz, robot_receive_timestamp
        for robot in self.robots:
            last_robots_data.append(robot.get_all_state())
        # 30 hz, gripper_receive_timestamp
        for gripper in self.grippers:
            last_grippers_data.append(gripper.get_all_state())

        # select align_camera_idx based on calibrated timestamps
        # The timestamps are already calibrated in UvcCamera, so we use them directly
        num_obs_cameras = len(self.v4l_paths)
        align_camera_idx = None
        running_best_error = np.inf
   
        for camera_idx in range(num_obs_cameras):
            this_error = 0
            this_timestamp = self.last_camera_data[camera_idx]['timestamp'][-1]
            for other_camera_idx in range(num_obs_cameras):
                if other_camera_idx == camera_idx:
                    continue
                other_timestep_idx = -1
                while True:
                    if self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx] < this_timestamp:
                        this_error += this_timestamp - self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx]
                        break
                    other_timestep_idx -= 1
            if align_camera_idx is None or this_error < running_best_error:
                running_best_error = this_error
                align_camera_idx = camera_idx

        last_timestamp = self.last_camera_data[align_camera_idx]['timestamp'][-1]
        dt = 1 / self.frequency

        # align camera obs timestamps
        # Since timestamps are already calibrated in UvcCamera, we can use them directly
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in camera_obs_timestamps:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                # Optional: Add warning for large timestamp mismatches
                # if np.abs(this_timestamps - t)[nn_idx] > 1.0 / 60:
                #     print(f'WARNING: Large timestamp mismatch for camera {camera_idx}: {np.abs(this_timestamps - t)[nn_idx]:.4f}s')
                this_idxs.append(nn_idx)
            # remap key
            if self.deploy_with_tactile:
                # because the left robot is robot_1, the right robot is robot_0
                if camera_idx == 0:
                    camera_obs['camera0_rgb'] = value['color'][...,:3][this_idxs]
                elif camera_idx == 1: #
                    if self.deploy_with_tactile_img:
                        tactile_frames = value['color'][...,:3][this_idxs]
                        camera_obs['camera0_left_tactile'] = tactile_frames
                    if self.deploy_with_tactile_pc:
                        # Get pre-computed tactile point clouds from camera thread
                        if 'tactile_points' in value:
                            tactile_points_data = value['tactile_points'][this_idxs]
                            # Stack into regular numpy array: (T, num_points, 3)
                            camera_obs['camera0_left_tactile_points'] = np.stack(tactile_points_data).astype(np.float32)
                        else:
                            # Fallback: create empty point cloud if data not available
                            T = len(this_idxs)
                            camera_obs['camera0_left_tactile_points'] = np.zeros((T, self.fps_num_points, 3), dtype=np.float32)

                elif camera_idx == 2:
                    if self.deploy_with_tactile_img:
                        tactile_frames = value['color'][...,:3][this_idxs]
                        camera_obs['camera0_right_tactile'] = tactile_frames
                    if self.deploy_with_tactile_pc:
                        # Get pre-computed tactile point clouds from camera thread
                        if 'tactile_points' in value:
                            tactile_points_data = value['tactile_points'][this_idxs]
                            # Stack into regular numpy array: (T, num_points, 3)
                            camera_obs['camera0_right_tactile_points'] = np.stack(tactile_points_data).astype(np.float32)
                        else:
                            # Fallback: create empty point cloud if data not available
                            T = len(this_idxs)
                            camera_obs['camera0_right_tactile_points'] = np.zeros((T, self.fps_num_points, 3), dtype=np.float32)

                elif camera_idx == 3:
                    camera_obs['camera1_rgb'] = value['color'][...,:3][this_idxs]

                elif camera_idx == 4:
                    if self.deploy_with_tactile_img:
                        tactile_frames = value['color'][...,:3][this_idxs]
                        camera_obs['camera1_left_tactile'] = tactile_frames
                    if self.deploy_with_tactile_pc:
                        # Get pre-computed tactile point clouds from camera thread
                        if 'tactile_points' in value:
                            tactile_points_data = value['tactile_points'][this_idxs]
                            # Stack into regular numpy array: (T, num_points, 3)
                            camera_obs['camera1_left_tactile_points'] = np.stack(tactile_points_data).astype(np.float32)
                        else:
                            # Fallback: create empty point cloud if data not available
                            T = len(this_idxs)
                            camera_obs['camera1_left_tactile_points'] = np.zeros((T, self.fps_num_points, 3), dtype=np.float32)

                elif camera_idx == 5:
                    if self.deploy_with_tactile_img:
                        tactile_frames = value['color'][...,:3][this_idxs]
                        camera_obs['camera1_right_tactile'] = tactile_frames
                    if self.deploy_with_tactile_pc:
                        # Get pre-computed tactile point clouds from camera thread
                        if 'tactile_points' in value:
                            tactile_points_data = value['tactile_points'][this_idxs]
                            # Stack into regular numpy array: (T, num_points, 3)
                            camera_obs['camera1_right_tactile_points'] = np.stack(tactile_points_data).astype(np.float32)
                        else:
                            # Fallback: create empty point cloud if data not available
                            T = len(this_idxs)
                            camera_obs['camera1_right_tactile_points'] = np.zeros((T, self.fps_num_points, 3), dtype=np.float32)
                else:
                    raise ValueError(f'Invalid camera index: {camera_idx}')
            else:
                if camera_idx == 0:
                    camera_obs['camera0_rgb'] = value['color'][..., :3][this_idxs]
                elif camera_idx == 1:
                    camera_obs['camera1_rgb'] = value['color'][..., :3][this_idxs]
                else:
                    raise ValueError(f'Invalid camera index: {camera_idx}')



        # obs_data to return (it only includes camera data at this stage)
        obs_data = dict(camera_obs)
        

        # include camera timesteps
        obs_data['timestamp'] = camera_obs_timestamps

        # align robot obs
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        for robot_idx, last_robot_data in enumerate(last_robots_data):
            robot_pose_interpolator = PoseInterpolator(
                t=last_robot_data['robot_timestamp'], 
                x=last_robot_data['ee_pose'])
            robot_pose = robot_pose_interpolator(robot_obs_timestamps)
            robot_obs = {
                f'robot{robot_idx}_eef_pos': robot_pose[...,:3],
                f'robot{robot_idx}_eef_rot_axis_angle': robot_pose[...,3:]
            }
            # update obs_data
            obs_data.update(robot_obs)

        # align gripper obs
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
        gripper_obs = {}
        for robot_idx, last_gripper_data in enumerate(last_grippers_data):
            # align gripper obs
            gripper_interpolator = get_interp1d(
                t=last_gripper_data['gripper_timestamp'],
                x=last_gripper_data['gripper_position'][...,None]
            )
            gripper_obs[f'robot{robot_idx}_gripper_width'] = gripper_interpolator(gripper_obs_timestamps)

        # update obs_data
        obs_data.update(gripper_obs)


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

        assert new_actions.shape[1] // len(self.robots) == 7
        assert new_actions.shape[1] % len(self.robots) == 0

        # schedule waypoints
        for i in range(len(new_actions)):
            for robot_idx, (robot, gripper) in enumerate(zip(self.robots, self.grippers)):
                r_latency = 0.0
                g_latency = 0.0
                r_actions = new_actions[i, 7 * robot_idx + 0: 7 * robot_idx + 6]
                if robot_idx == 1:
                    g_actions = new_actions[i, 7 * robot_idx + 6]
                else:
                    g_actions = new_actions[i, 7 * robot_idx + 6]

                robot.schedule_waypoint(
                    pose=r_actions,
                    target_time=new_timestamps[i] - r_latency
                )
                gripper.schedule_waypoint(
                    pos=g_actions,
                    target_time=new_timestamps[i] - g_latency
                )
    
    def get_robot_state(self):
        return [robot.get_state() for robot in self.robots]
    
    def get_gripper_state(self):
        return [gripper.get_state() for gripper in self.grippers]

    