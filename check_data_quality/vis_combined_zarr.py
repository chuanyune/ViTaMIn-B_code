import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import cv2
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
register_codecs()

def get_transformation_matrix(pos, rot_axis_angle):
    rotation_matrix, _ = cv2.Rodrigues(rot_axis_angle)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = pos
    return transformation_matrix

def create_bimanual_combined_image(visual_1, left_tactile_1, right_tactile_1, 
                                   visual_0, left_tactile_0, right_tactile_0,
                                   left_pc_image_0, right_pc_image_0, 
                                   left_pc_image_1, right_pc_image_1,
                                   traj_image_0, traj_image_1):

    target_height = 250
    
    robot0_traj = None
    robot0_pc_section = []
    robot0_images = []
    robot0_labels = []
    
    if traj_image_0 is not None:
        h, w = traj_image_0.shape[:2]
        target_width = int(w * target_height / h)
        robot0_traj = cv2.resize(traj_image_0, (target_width, target_height))
        cv2.putText(robot0_traj, "R0 Trajectory", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(robot0_traj, "R0 Trajectory", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    if left_pc_image_0 is not None:
        h, w = left_pc_image_0.shape[:2]
        target_width = int(w * target_height / h)
        resized = cv2.resize(left_pc_image_0, (target_width, target_height))
        cv2.putText(resized, "R0 Left PC", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(resized, "R0 Left PC", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        robot0_pc_section.append(resized)
    
    if right_pc_image_0 is not None:
        h, w = right_pc_image_0.shape[:2]
        target_width = int(w * target_height / h)
        resized = cv2.resize(right_pc_image_0, (target_width, target_height))
        cv2.putText(resized, "R0 Right PC", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(resized, "R0 Right PC", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        robot0_pc_section.append(resized)
    
    if visual_0 is not None:
        robot0_images.append(visual_0)
        robot0_labels.append("R0 Visual")
    if left_tactile_0 is not None:
        robot0_images.append(left_tactile_0)
        robot0_labels.append("R0 L-Tact")
    if right_tactile_0 is not None:
        robot0_images.append(right_tactile_0)
        robot0_labels.append("R0 R-Tact")
    
    robot1_traj = None
    robot1_pc_section = []
    robot1_images = []
    robot1_labels = []
    
    if traj_image_1 is not None:
        h, w = traj_image_1.shape[:2]
        target_width = int(w * target_height / h)
        robot1_traj = cv2.resize(traj_image_1, (target_width, target_height))
        cv2.putText(robot1_traj, "R1 Trajectory", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(robot1_traj, "R1 Trajectory", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    if left_pc_image_1 is not None:
        h, w = left_pc_image_1.shape[:2]
        target_width = int(w * target_height / h)
        resized = cv2.resize(left_pc_image_1, (target_width, target_height))
        cv2.putText(resized, "R1 Left PC", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(resized, "R1 Left PC", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        robot1_pc_section.append(resized)
    
    if right_pc_image_1 is not None:
        h, w = right_pc_image_1.shape[:2]
        target_width = int(w * target_height / h)
        resized = cv2.resize(right_pc_image_1, (target_width, target_height))
        cv2.putText(resized, "R1 Right PC", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(resized, "R1 Right PC", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        robot1_pc_section.append(resized)
    
    if visual_1 is not None:
        robot1_images.append(visual_1)
        robot1_labels.append("R1 Visual")
    if left_tactile_1 is not None:
        robot1_images.append(left_tactile_1)
        robot1_labels.append("R1 L-Tact")
    if right_tactile_1 is not None:
        robot1_images.append(right_tactile_1)
        robot1_labels.append("R1 R-Tact")
    
    robot0_row = None
    if robot0_traj is not None or robot0_pc_section or robot0_images:
        row_components = []
        
        if robot0_traj is not None:
            row_components.append(robot0_traj)
        
        if robot0_pc_section:
            pc_part = np.hstack(robot0_pc_section)
            x_offset = 0
            for i in range(len(robot0_pc_section) - 1):
                x_offset += robot0_pc_section[i].shape[1]
                cv2.line(pc_part, (x_offset, 0), (x_offset, target_height), (255, 255, 255), 2)
            row_components.append(pc_part)
        
        if robot0_images:
            resized_imgs = []
            for img in robot0_images:
                h, w = img.shape[:2]
                target_width = int(w * target_height / h)
                resized_img = cv2.resize(img, (target_width, target_height))
                resized_imgs.append(resized_img)
            
            img_part = np.hstack(resized_imgs)
            
            x_offset = 0
            for i, (img, label) in enumerate(zip(resized_imgs, robot0_labels)):
                cv2.putText(img_part, label, (x_offset + 10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img_part, label, (x_offset + 10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
                if i < len(resized_imgs) - 1:
                    x_offset += img.shape[1]
                    cv2.line(img_part, (x_offset, 0), (x_offset, target_height), (255, 255, 255), 2)
                else:
                    x_offset += img.shape[1]
            
            row_components.append(img_part)
        
        if len(row_components) > 1:
            separator = np.ones((target_height, 5, 3), dtype=np.uint8) * 128
            parts_with_sep = []
            for i, comp in enumerate(row_components):
                parts_with_sep.append(comp)
                if i < len(row_components) - 1:
                    parts_with_sep.append(separator)
            robot0_row = np.hstack(parts_with_sep)
        elif len(row_components) == 1:
            robot0_row = row_components[0]
    
    robot1_row = None
    if robot1_traj is not None or robot1_pc_section or robot1_images:
        row_components = []
        
        if robot1_traj is not None:
            row_components.append(robot1_traj)
        
        if robot1_pc_section:
            pc_part = np.hstack(robot1_pc_section)
            x_offset = 0
            for i in range(len(robot1_pc_section) - 1):
                x_offset += robot1_pc_section[i].shape[1]
                cv2.line(pc_part, (x_offset, 0), (x_offset, target_height), (255, 255, 255), 2)
            row_components.append(pc_part)
        
        if robot1_images:
            resized_imgs = []
            for img in robot1_images:
                h, w = img.shape[:2]
                target_width = int(w * target_height / h)
                resized_img = cv2.resize(img, (target_width, target_height))
                resized_imgs.append(resized_img)
            
            img_part = np.hstack(resized_imgs)
            
            x_offset = 0
            for i, (img, label) in enumerate(zip(resized_imgs, robot1_labels)):
                cv2.putText(img_part, label, (x_offset + 10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img_part, label, (x_offset + 10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                if i < len(resized_imgs) - 1:
                    x_offset += img.shape[1]
                    cv2.line(img_part, (x_offset, 0), (x_offset, target_height), (255, 255, 255), 2)
                else:
                    x_offset += img.shape[1]
            
            row_components.append(img_part)
        
        if len(row_components) > 1:
            separator = np.ones((target_height, 5, 3), dtype=np.uint8) * 128
            parts_with_sep = []
            for i, comp in enumerate(row_components):
                parts_with_sep.append(comp)
                if i < len(row_components) - 1:
                    parts_with_sep.append(separator)
            robot1_row = np.hstack(parts_with_sep)
        elif len(row_components) == 1:
            robot1_row = row_components[0]
    
    if robot0_row is not None and robot1_row is not None:
        max_width = max(robot0_row.shape[1], robot1_row.shape[1])
        
        if robot0_row.shape[1] < max_width:
            padding = np.zeros((target_height, max_width - robot0_row.shape[1], 3), dtype=np.uint8)
            robot0_row = np.hstack([robot0_row, padding])
        
        if robot1_row.shape[1] < max_width:
            padding = np.zeros((target_height, max_width - robot1_row.shape[1], 3), dtype=np.uint8)
            robot1_row = np.hstack([robot1_row, padding])
        
        separator = np.ones((5, max_width, 3), dtype=np.uint8) * 128
        combined_image = np.vstack([robot0_row, separator, robot1_row])
        
    elif robot0_row is not None:
        combined_image = robot0_row
    elif robot1_row is not None:
        combined_image = robot1_row
    else:
        return None
    
    return combined_image

def load_episode_data(replay_buffer, episode_idx):
    """Load episode data including images and point clouds"""
    episode_slice = replay_buffer.get_episode_slice(episode_idx)
    start_idx = episode_slice.start
    stop_idx = episode_slice.stop

    pos_list_0 = []
    pos_list_1 = []
    
    visual_images_0 = []
    visual_images_1 = []
    left_tactile_images_0 = []
    right_tactile_images_0 = []
    left_tactile_images_1 = []
    right_tactile_images_1 = []
    
    left_tactile_pointclouds_0 = []
    right_tactile_pointclouds_0 = []
    left_tactile_pointclouds_1 = []
    right_tactile_pointclouds_1 = []
    
    gripper_widths_0 = []
    gripper_widths_1 = []
    first_frame_tx_0 = None
    first_frame_tx_1 = None

    has_visual_0 = 'camera0_rgb' in replay_buffer.keys()
    has_visual_1 = 'camera1_rgb' in replay_buffer.keys()
    has_left_tactile_0 = 'camera0_left_tactile' in replay_buffer.keys()
    has_right_tactile_0 = 'camera0_right_tactile' in replay_buffer.keys()
    has_left_tactile_1 = 'camera1_left_tactile' in replay_buffer.keys()
    has_right_tactile_1 = 'camera1_right_tactile' in replay_buffer.keys()
    has_robot1_data = 'robot1_eef_pos' in replay_buffer.keys()
    has_gripper0_data = 'robot0_gripper_width' in replay_buffer.keys()
    has_gripper1_data = 'robot1_gripper_width' in replay_buffer.keys()
    
    has_left_tactile_pc_0 = 'camera0_left_tactile_points' in replay_buffer.keys()
    has_right_tactile_pc_0 = 'camera0_right_tactile_points' in replay_buffer.keys()
    has_left_tactile_pc_1 = 'camera1_left_tactile_points' in replay_buffer.keys()
    has_right_tactile_pc_1 = 'camera1_right_tactile_points' in replay_buffer.keys()
    
    print(f"📊 Data Check Results:")
    print(f"  Robot0 (Camera0):")
    print(f"    Visual RGB: {'✅' if has_visual_0 else '❌'}")
    print(f"    Left Tactile Image: {'✅' if has_left_tactile_0 else '❌'}")
    print(f"    Right Tactile Image: {'✅' if has_right_tactile_0 else '❌'}")
    print(f"    Left Tactile PC: {'✅' if has_left_tactile_pc_0 else '❌'}")
    print(f"    Right Tactile PC: {'✅' if has_right_tactile_pc_0 else '❌'}")
    print(f"    Gripper Width: {'✅' if has_gripper0_data else '❌'}")
    print(f"  Robot1 (Camera1):")
    print(f"    Visual RGB: {'✅' if has_visual_1 else '❌'}")
    print(f"    Left Tactile Image: {'✅' if has_left_tactile_1 else '❌'}")
    print(f"    Right Tactile Image: {'✅' if has_right_tactile_1 else '❌'}")
    print(f"    Left Tactile PC: {'✅' if has_left_tactile_pc_1 else '❌'}")
    print(f"    Right Tactile PC: {'✅' if has_right_tactile_pc_1 else '❌'}")
    print(f"    Gripper Width: {'✅' if has_gripper1_data else '❌'}")
    print(f"    Robot1 Pose: {'✅' if has_robot1_data else '❌'}")

    for i in range(start_idx, stop_idx, 1):
        pos_0 = replay_buffer['robot0_eef_pos'][i]
        rot_0 = replay_buffer['robot0_eef_rot_axis_angle'][i]
        
        if has_gripper0_data:
            gripper_width_0 = replay_buffer['robot0_gripper_width'][i][0]
            gripper_widths_0.append(gripper_width_0)
        
        if has_visual_0:
            img_data = replay_buffer['camera0_rgb'][i]
            if isinstance(img_data, bytes):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img_data
            visual_images_0.append(img)
        
        if has_left_tactile_0:
            img_data = replay_buffer['camera0_left_tactile'][i]
            if isinstance(img_data, bytes):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img_data
            left_tactile_images_0.append(img)
        
        if has_right_tactile_0:
            img_data = replay_buffer['camera0_right_tactile'][i]
            if isinstance(img_data, bytes):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img_data
            right_tactile_images_0.append(img)
        
        if has_left_tactile_pc_0:
            try:
                points_data = replay_buffer['camera0_left_tactile_points'][i]
                if points_data is not None and len(points_data) > 0:
                    if isinstance(points_data, (list, tuple)):
                        points_array = np.array(points_data, dtype=np.float32)
                    else:
                        points_array = np.array(points_data, dtype=np.float32)
                    
                    if points_array.ndim == 2 and points_array.shape[1] == 3:
                        non_zero_mask = np.any(points_array != 0, axis=1)
                        if np.any(non_zero_mask):
                            filtered_points = points_array[non_zero_mask]
                            left_tactile_pointclouds_0.append(filtered_points)
                        else:
                            left_tactile_pointclouds_0.append(np.empty((0, 3), dtype=np.float32))
                    else:
                        left_tactile_pointclouds_0.append(np.empty((0, 3), dtype=np.float32))
                else:
                    left_tactile_pointclouds_0.append(np.empty((0, 3), dtype=np.float32))
            except Exception as e:
                left_tactile_pointclouds_0.append(np.empty((0, 3), dtype=np.float32))
        
        if has_right_tactile_pc_0:
            try:
                points_data = replay_buffer['camera0_right_tactile_points'][i]
                if points_data is not None and len(points_data) > 0:
                    if isinstance(points_data, (list, tuple)):
                        points_array = np.array(points_data, dtype=np.float32)
                    else:
                        points_array = np.array(points_data, dtype=np.float32)
                    
                    if points_array.ndim == 2 and points_array.shape[1] == 3:
                        non_zero_mask = np.any(points_array != 0, axis=1)
                        if np.any(non_zero_mask):
                            filtered_points = points_array[non_zero_mask]
                            right_tactile_pointclouds_0.append(filtered_points)
                        else:
                            right_tactile_pointclouds_0.append(np.empty((0, 3), dtype=np.float32))
                    else:
                        right_tactile_pointclouds_0.append(np.empty((0, 3), dtype=np.float32))
                else:
                    right_tactile_pointclouds_0.append(np.empty((0, 3), dtype=np.float32))
            except Exception as e:
                right_tactile_pointclouds_0.append(np.empty((0, 3), dtype=np.float32))
        
        if has_robot1_data:
            pos_1 = replay_buffer['robot1_eef_pos'][i]
            rot_1 = replay_buffer['robot1_eef_rot_axis_angle'][i]
            
            if has_gripper1_data:
                gripper_width_1 = replay_buffer['robot1_gripper_width'][i][0]
                gripper_widths_1.append(gripper_width_1)
            
            transform_1 = get_transformation_matrix(pos_1, rot_1)
            
            if first_frame_tx_1 is None:
                first_frame_tx_1 = transform_1.copy()
            
            rel_transform_1 = np.linalg.inv(first_frame_tx_1) @ transform_1
            pos_list_1.append(rel_transform_1)
        
        if has_visual_1:
            img_data = replay_buffer['camera1_rgb'][i]
            if isinstance(img_data, bytes):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img_data
            visual_images_1.append(img)
        
        if has_left_tactile_1:
            img_data = replay_buffer['camera1_left_tactile'][i]
            if isinstance(img_data, bytes):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img_data
            left_tactile_images_1.append(img)
        
        if has_right_tactile_1:
            img_data = replay_buffer['camera1_right_tactile'][i]
            if isinstance(img_data, bytes):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img_data
            right_tactile_images_1.append(img)
        
        if has_left_tactile_pc_1:
            try:
                points_data = replay_buffer['camera1_left_tactile_points'][i]
                if points_data is not None and len(points_data) > 0:
                    if isinstance(points_data, (list, tuple)):
                        points_array = np.array(points_data, dtype=np.float32)
                    else:
                        points_array = np.array(points_data, dtype=np.float32)
                    
                    if points_array.ndim == 2 and points_array.shape[1] == 3:
                        non_zero_mask = np.any(points_array != 0, axis=1)
                        if np.any(non_zero_mask):
                            filtered_points = points_array[non_zero_mask]
                            left_tactile_pointclouds_1.append(filtered_points)
                        else:
                            left_tactile_pointclouds_1.append(np.empty((0, 3), dtype=np.float32))
                    else:
                        left_tactile_pointclouds_1.append(np.empty((0, 3), dtype=np.float32))
                else:
                    left_tactile_pointclouds_1.append(np.empty((0, 3), dtype=np.float32))
            except Exception as e:
                left_tactile_pointclouds_1.append(np.empty((0, 3), dtype=np.float32))
        
        if has_right_tactile_pc_1:
            try:
                points_data = replay_buffer['camera1_right_tactile_points'][i]
                if points_data is not None and len(points_data) > 0:
                    if isinstance(points_data, (list, tuple)):
                        points_array = np.array(points_data, dtype=np.float32)
                    else:
                        points_array = np.array(points_data, dtype=np.float32)
                    
                    if points_array.ndim == 2 and points_array.shape[1] == 3:
                        non_zero_mask = np.any(points_array != 0, axis=1)
                        if np.any(non_zero_mask):
                            filtered_points = points_array[non_zero_mask]
                            right_tactile_pointclouds_1.append(filtered_points)
                        else:
                            right_tactile_pointclouds_1.append(np.empty((0, 3), dtype=np.float32))
                    else:
                        right_tactile_pointclouds_1.append(np.empty((0, 3), dtype=np.float32))
                else:
                    right_tactile_pointclouds_1.append(np.empty((0, 3), dtype=np.float32))
            except Exception as e:
                right_tactile_pointclouds_1.append(np.empty((0, 3), dtype=np.float32))
        
        transform_0 = get_transformation_matrix(pos_0, rot_0)
        
        if first_frame_tx_0 is None:
            first_frame_tx_0 = transform_0.copy()
        
        rel_transform_0 = np.linalg.inv(first_frame_tx_0) @ transform_0
        pos_list_0.append(rel_transform_0)

    return (pos_list_0, pos_list_1, 
            visual_images_0, visual_images_1, left_tactile_images_0, right_tactile_images_0,
            left_tactile_images_1, right_tactile_images_1, 
            left_tactile_pointclouds_0, right_tactile_pointclouds_0,
            left_tactile_pointclouds_1, right_tactile_pointclouds_1,
            gripper_widths_0, gripper_widths_1,
            has_visual_0, has_visual_1, has_left_tactile_0, has_right_tactile_0,
            has_left_tactile_1, has_right_tactile_1, has_robot1_data, 
            has_gripper0_data, has_gripper1_data, 
            has_left_tactile_pc_0, has_right_tactile_pc_0,
            has_left_tactile_pc_1, has_right_tactile_pc_1)

class CombinedVisualizer:
    def __init__(self, replay_buffer, available_episodes, 
                 record_mode=False, record_episode=0, output_video=None, 
                 record_fps=30, continue_after_record=False):
        self.replay_buffer = replay_buffer
        self.available_episodes = available_episodes
        self.current_episode_idx = 0
        self.current_frame_idx = 0
        
        self.record_mode = record_mode
        self.record_episode = record_episode
        self.output_video = output_video
        self.record_fps = record_fps
        self.continue_after_record = continue_after_record
        
        if self.record_mode:
            self.current_episode_idx = self.record_episode
        
        self.load_current_episode()
        
        self.setup_offscreen_renderers()
        
        if self.record_mode:
            self.record_current_episode()
            if not self.continue_after_record:
                print("\n✅ Recording complete, exiting program")
                return
            else:
                print("\n✅ Recording complete, entering interactive mode")
        
        self.print_help()
        
        self.run()
    
    def load_current_episode(self):
        """Load current episode data"""
        episode_id = self.available_episodes[self.current_episode_idx]
        
        (self.pos_list_0, self.pos_list_1, 
         self.visual_images_0, self.visual_images_1, 
         self.left_tactile_images_0, self.right_tactile_images_0,
         self.left_tactile_images_1, self.right_tactile_images_1,
         self.left_tactile_pointclouds_0, self.right_tactile_pointclouds_0,
         self.left_tactile_pointclouds_1, self.right_tactile_pointclouds_1,
         self.gripper_widths_0, self.gripper_widths_1,
         self.has_visual_0, self.has_visual_1, 
         self.has_left_tactile_0, self.has_right_tactile_0,
         self.has_left_tactile_1, self.has_right_tactile_1, 
         self.has_robot1_data, self.has_gripper0_data, self.has_gripper1_data,
         self.has_left_tactile_pc_0, self.has_right_tactile_pc_0,
         self.has_left_tactile_pc_1, self.has_right_tactile_pc_1) = load_episode_data(self.replay_buffer, episode_id)
        
        self.current_frame_idx = 0
        
        self.setup_fixed_camera_params()
        
        print(f"\n✅ Loaded Episode {episode_id} ({self.current_episode_idx + 1}/{len(self.available_episodes)}):")
        print(f"   Total frames: {len(self.pos_list_0)}")
        if self.has_left_tactile_pc_0:
            valid_frames = sum(1 for pc in self.left_tactile_pointclouds_0 if len(pc) > 0)
            total_points = sum(len(pc) for pc in self.left_tactile_pointclouds_0)
            print(f"   Robot0 left tactile PC: {valid_frames}/{len(self.left_tactile_pointclouds_0)} valid frames, {total_points} points total")
        if self.has_right_tactile_pc_0:
            valid_frames = sum(1 for pc in self.right_tactile_pointclouds_0 if len(pc) > 0)
            total_points = sum(len(pc) for pc in self.right_tactile_pointclouds_0)
            print(f"   Robot0 right tactile PC: {valid_frames}/{len(self.right_tactile_pointclouds_0)} valid frames, {total_points} points total")
        if self.has_left_tactile_pc_1:
            valid_frames = sum(1 for pc in self.left_tactile_pointclouds_1 if len(pc) > 0)
            total_points = sum(len(pc) for pc in self.left_tactile_pointclouds_1)
            print(f"   Robot1 left tactile PC: {valid_frames}/{len(self.left_tactile_pointclouds_1)} valid frames, {total_points} points total")
        if self.has_right_tactile_pc_1:
            valid_frames = sum(1 for pc in self.right_tactile_pointclouds_1 if len(pc) > 0)
            total_points = sum(len(pc) for pc in self.right_tactile_pointclouds_1)
            print(f"   Robot1 right tactile PC: {valid_frames}/{len(self.right_tactile_pointclouds_1)} valid frames, {total_points} points total")
    
    def setup_fixed_camera_params(self):
        """Calculate and fix camera parameters for 4 point clouds based on first frame"""
        default_center = [0, 0, 40]
        default_eye = [50, -30, 80]
        default_up = [0, 0, 1]
        
        self.left_camera_params_0 = {'center': default_center, 'eye': default_eye, 'up': default_up}
        self.right_camera_params_0 = {'center': default_center, 'eye': default_eye, 'up': default_up}
        self.left_camera_params_1 = {'center': default_center, 'eye': default_eye, 'up': default_up}
        self.right_camera_params_1 = {'center': default_center, 'eye': default_eye, 'up': default_up}
        
        if self.has_left_tactile_pc_0 and len(self.left_tactile_pointclouds_0) > 0:
            for points in self.left_tactile_pointclouds_0:
                if len(points) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    center = pcd.get_center()
                    bounds = pcd.get_axis_aligned_bounding_box()
                    extent = bounds.get_extent()
                    max_extent = np.max(extent)
                    distance = max_extent * 2.5
                    eye = center + np.array([distance * 0.5, -distance * 0.3, distance * 0.8])
                    self.left_camera_params_0 = {'center': center.tolist(), 'eye': eye.tolist(), 'up': [0, 0, 1]}
                    break
        
        if self.has_right_tactile_pc_0 and len(self.right_tactile_pointclouds_0) > 0:
            for points in self.right_tactile_pointclouds_0:
                if len(points) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    center = pcd.get_center()
                    bounds = pcd.get_axis_aligned_bounding_box()
                    extent = bounds.get_extent()
                    max_extent = np.max(extent)
                    distance = max_extent * 2.5
                    eye = center + np.array([distance * 0.5, -distance * 0.3, distance * 0.8])
                    self.right_camera_params_0 = {'center': center.tolist(), 'eye': eye.tolist(), 'up': [0, 0, 1]}
                    break
        
        if self.has_left_tactile_pc_1 and len(self.left_tactile_pointclouds_1) > 0:
            for points in self.left_tactile_pointclouds_1:
                if len(points) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    center = pcd.get_center()
                    bounds = pcd.get_axis_aligned_bounding_box()
                    extent = bounds.get_extent()
                    max_extent = np.max(extent)
                    distance = max_extent * 2.5
                    eye = center + np.array([distance * 0.5, -distance * 0.3, distance * 0.8])
                    self.left_camera_params_1 = {'center': center.tolist(), 'eye': eye.tolist(), 'up': [0, 0, 1]}
                    break
        
        if self.has_right_tactile_pc_1 and len(self.right_tactile_pointclouds_1) > 0:
            for points in self.right_tactile_pointclouds_1:
                if len(points) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    center = pcd.get_center()
                    bounds = pcd.get_axis_aligned_bounding_box()
                    extent = bounds.get_extent()
                    max_extent = np.max(extent)
                    distance = max_extent * 2.5
                    eye = center + np.array([distance * 0.5, -distance * 0.3, distance * 0.8])
                    self.right_camera_params_1 = {'center': center.tolist(), 'eye': eye.tolist(), 'up': [0, 0, 1]}
                    break
        
        print(f"📷 Fixed camera parameters set (4 point clouds, based on first frame)")
    
    def setup_offscreen_renderers(self):
        """Setup Open3D offscreen renderers (6 renderers: 4 tactile point clouds + 2 trajectories)"""
        self.render_width = 400
        self.render_height = 300
        
        self.vis_left_0 = o3d.visualization.rendering.OffscreenRenderer(self.render_width, self.render_height)
        self.vis_right_0 = o3d.visualization.rendering.OffscreenRenderer(self.render_width, self.render_height)
        self.vis_left_1 = o3d.visualization.rendering.OffscreenRenderer(self.render_width, self.render_height)
        self.vis_right_1 = o3d.visualization.rendering.OffscreenRenderer(self.render_width, self.render_height)
        self.vis_traj_0 = o3d.visualization.rendering.OffscreenRenderer(self.render_width, self.render_height)
        self.vis_traj_1 = o3d.visualization.rendering.OffscreenRenderer(self.render_width, self.render_height)
        
        all_renderers = [self.vis_left_0, self.vis_right_0, self.vis_left_1, self.vis_right_1, 
                        self.vis_traj_0, self.vis_traj_1]
        for vis in all_renderers:
            vis.scene.set_background([0.1, 0.1, 0.1, 1.0])
            vis.scene.scene.set_sun_light([0, -1, -1], [1.0, 1.0, 1.0], 50000)
            vis.scene.scene.enable_sun_light(True)
        
        print(f"✅ Initialized offscreen renderers (6 renderers: 4 tactile PC + 2 trajectories): {self.render_width}x{self.render_height}")
    
    def render_pointcloud(self, points, renderer, name="pointcloud", camera_params=None):
        """Render point cloud to image using fixed camera parameters"""
        renderer.scene.clear_geometry()
        
        pcd_mat = o3d.visualization.rendering.MaterialRecord()
        pcd_mat.shader = 'defaultUnlit'
        pcd_mat.point_size = 8.0
        
        if len(points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            colors = np.tile([1.0, 1.0, 0.0], (len(points), 1))
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            renderer.scene.add_geometry(name, pcd, pcd_mat)
        
        if camera_params is not None:
            renderer.setup_camera(60.0, 
                                camera_params['center'], 
                                camera_params['eye'], 
                                camera_params['up'])
        else:
            renderer.setup_camera(60.0, [0, 0, 40], [50, -30, 80], [0, 0, 1])
        
        coord_mat = o3d.visualization.rendering.MaterialRecord()
        coord_mat.shader = 'defaultUnlit'
        coord_mat.line_width = 2.0
        
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
        renderer.scene.add_geometry("coordinate", coord_frame, coord_mat)
        
        img = renderer.render_to_image()
        
        img_np = np.asarray(img)
        
        return img_np
    
    def render_trajectory(self, pos_list, current_idx, renderer, name="trajectory", color=[1, 0, 0]):
        """Render trajectory to image showing trajectory up to current frame"""
        renderer.scene.clear_geometry()
        
        if len(pos_list) == 0 or current_idx >= len(pos_list):
            renderer.setup_camera(60.0, [0, 0, 0], [0.5, -0.3, 0.8], [0, 0, 1])
            img = renderer.render_to_image()
            return np.asarray(img)
        
        line_mat = o3d.visualization.rendering.MaterialRecord()
        line_mat.shader = 'defaultUnlit'
        line_mat.line_width = 3.0
        
        point_mat = o3d.visualization.rendering.MaterialRecord()
        point_mat.shader = 'defaultUnlit'
        point_mat.point_size = 8.0
        
        frame_mat = o3d.visualization.rendering.MaterialRecord()
        frame_mat.shader = 'defaultUnlit'
        
        trajectory_points = np.array([pos[:3, 3] for pos in pos_list[:current_idx+1]])
        
        if len(trajectory_points) > 1:
            lines = [[i, i+1] for i in range(len(trajectory_points)-1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(trajectory_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
            renderer.scene.add_geometry(f"{name}_lines", line_set, line_mat)
        
        if len(trajectory_points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(trajectory_points)
            pcd.colors = o3d.utility.Vector3dVector([color for _ in range(len(trajectory_points))])
            renderer.scene.add_geometry(f"{name}_points", pcd, point_mat)
        
        curr_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        curr_frame.transform(pos_list[current_idx])
        renderer.scene.add_geometry(f"{name}_current_frame", curr_frame, frame_mat)
        
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        renderer.scene.add_geometry(f"{name}_origin", origin_frame, frame_mat)
        
        if len(trajectory_points) > 0:
            center = trajectory_points.mean(axis=0)
            extent = trajectory_points.max(axis=0) - trajectory_points.min(axis=0)
            max_extent = np.max(extent)
            
            distance = max(max_extent * 2.0, 0.3)
            eye = center + np.array([distance * 0.6, -distance * 0.4, distance * 0.8])
            up = [0, 0, 1]
            
            renderer.setup_camera(60.0, center.tolist(), eye.tolist(), up)
        else:
            renderer.setup_camera(60.0, [0, 0, 0], [0.5, -0.3, 0.8], [0, 0, 1])
        
        img = renderer.render_to_image()
        img_np = np.asarray(img)
        
        return img_np
    
    def get_current_frame_images(self):
        """Get all images for the current frame"""
        visual_1 = None
        left_tactile_1 = None
        right_tactile_1 = None
        visual_0 = None
        left_tactile_0 = None
        right_tactile_0 = None
        
        if self.has_visual_1 and self.current_frame_idx < len(self.visual_images_1):
            visual_1 = self.visual_images_1[self.current_frame_idx].copy()
        if self.has_left_tactile_1 and self.current_frame_idx < len(self.left_tactile_images_1):
            left_tactile_1 = self.left_tactile_images_1[self.current_frame_idx].copy()
        if self.has_right_tactile_1 and self.current_frame_idx < len(self.right_tactile_images_1):
            right_tactile_1 = self.right_tactile_images_1[self.current_frame_idx].copy()
            
        if self.has_visual_0 and self.current_frame_idx < len(self.visual_images_0):
            visual_0 = self.visual_images_0[self.current_frame_idx].copy()
        if self.has_left_tactile_0 and self.current_frame_idx < len(self.left_tactile_images_0):
            left_tactile_0 = self.left_tactile_images_0[self.current_frame_idx].copy()
        if self.has_right_tactile_0 and self.current_frame_idx < len(self.right_tactile_images_0):
            right_tactile_0 = self.right_tactile_images_0[self.current_frame_idx].copy()
        
        left_pc_image_0 = None
        right_pc_image_0 = None
        left_pc_image_1 = None
        right_pc_image_1 = None
        
        if self.has_left_tactile_pc_0 and self.current_frame_idx < len(self.left_tactile_pointclouds_0):
            points = self.left_tactile_pointclouds_0[self.current_frame_idx]
            left_pc_image_0 = self.render_pointcloud(points, self.vis_left_0, "left_pc_0", self.left_camera_params_0)
        
        if self.has_right_tactile_pc_0 and self.current_frame_idx < len(self.right_tactile_pointclouds_0):
            points = self.right_tactile_pointclouds_0[self.current_frame_idx]
            right_pc_image_0 = self.render_pointcloud(points, self.vis_right_0, "right_pc_0", self.right_camera_params_0)
        
        if self.has_left_tactile_pc_1 and self.current_frame_idx < len(self.left_tactile_pointclouds_1):
            points = self.left_tactile_pointclouds_1[self.current_frame_idx]
            left_pc_image_1 = self.render_pointcloud(points, self.vis_left_1, "left_pc_1", self.left_camera_params_1)
        
        if self.has_right_tactile_pc_1 and self.current_frame_idx < len(self.right_tactile_pointclouds_1):
            points = self.right_tactile_pointclouds_1[self.current_frame_idx]
            right_pc_image_1 = self.render_pointcloud(points, self.vis_right_1, "right_pc_1", self.right_camera_params_1)
        
        traj_image_0 = None
        traj_image_1 = None
        
        if len(self.pos_list_0) > 0:
            traj_image_0 = self.render_trajectory(self.pos_list_0, self.current_frame_idx, 
                                                  self.vis_traj_0, "traj_0", color=[1, 0, 0])
        
        if self.has_robot1_data and len(self.pos_list_1) > 0:
            traj_image_1 = self.render_trajectory(self.pos_list_1, self.current_frame_idx, 
                                                  self.vis_traj_1, "traj_1", color=[0, 1, 0])
        
        return (visual_1, left_tactile_1, right_tactile_1, visual_0, left_tactile_0, right_tactile_0, 
                left_pc_image_0, right_pc_image_0, left_pc_image_1, right_pc_image_1,
                traj_image_0, traj_image_1)
    
    def update_display(self):
        """Update display"""
        (visual_1, left_tactile_1, right_tactile_1, visual_0, left_tactile_0, right_tactile_0, 
         left_pc_image_0, right_pc_image_0, left_pc_image_1, right_pc_image_1,
         traj_image_0, traj_image_1) = self.get_current_frame_images()
        
        combined_image = create_bimanual_combined_image(
            visual_1, left_tactile_1, right_tactile_1,
            visual_0, left_tactile_0, right_tactile_0,
            left_pc_image_0, right_pc_image_0, left_pc_image_1, right_pc_image_1,
            traj_image_0, traj_image_1
        )
        
        if combined_image is not None:
            img_height, img_width = combined_image.shape[:2]
            info_bar_height = 120
            
            info_bar = np.zeros((info_bar_height, img_width, 3), dtype=np.uint8)
            info_bar[:, :] = [30, 30, 30]
            
            episode_id = self.available_episodes[self.current_episode_idx]
            max_frames = len(self.pos_list_0)
            
            info_lines = [
                f"Episode {episode_id} ({self.current_episode_idx + 1}/{len(self.available_episodes)}) | Frame {self.current_frame_idx}/{max_frames-1}",
            ]
            
            current_pos_0 = self.pos_list_0[self.current_frame_idx][:3, 3]
            info_lines.append(f"Robot0 Pose: [{current_pos_0[0]:.3f}, {current_pos_0[1]:.3f}, {current_pos_0[2]:.3f}]")
            
            if self.has_robot1_data and self.current_frame_idx < len(self.pos_list_1):
                current_pos_1 = self.pos_list_1[self.current_frame_idx][:3, 3]
                info_lines.append(f"Robot1 Pose: [{current_pos_1[0]:.3f}, {current_pos_1[1]:.3f}, {current_pos_1[2]:.3f}]")
            
            if self.has_gripper0_data and self.current_frame_idx < len(self.gripper_widths_0):
                gripper_width_0 = self.gripper_widths_0[self.current_frame_idx]
                info_lines.append(f"Robot0 Gripper: {gripper_width_0:.4f}m")
            if self.has_gripper1_data and self.current_frame_idx < len(self.gripper_widths_1):
                gripper_width_1 = self.gripper_widths_1[self.current_frame_idx]
                info_lines.append(f"Robot1 Gripper: {gripper_width_1:.4f}m")
            
            pc_info = []
            if self.has_left_tactile_pc_0 and self.current_frame_idx < len(self.left_tactile_pointclouds_0):
                n_points = len(self.left_tactile_pointclouds_0[self.current_frame_idx])
                pc_info.append(f"R0-L:{n_points}")
            if self.has_right_tactile_pc_0 and self.current_frame_idx < len(self.right_tactile_pointclouds_0):
                n_points = len(self.right_tactile_pointclouds_0[self.current_frame_idx])
                pc_info.append(f"R0-R:{n_points}")
            if self.has_left_tactile_pc_1 and self.current_frame_idx < len(self.left_tactile_pointclouds_1):
                n_points = len(self.left_tactile_pointclouds_1[self.current_frame_idx])
                pc_info.append(f"R1-L:{n_points}")
            if self.has_right_tactile_pc_1 and self.current_frame_idx < len(self.right_tactile_pointclouds_1):
                n_points = len(self.right_tactile_pointclouds_1[self.current_frame_idx])
                pc_info.append(f"R1-R:{n_points}")
            if pc_info:
                info_lines.append(f"Point Clouds: {' | '.join(pc_info)}")
            
            y_offset = 20
            for line in info_lines:
                cv2.putText(info_bar, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                y_offset += 18
            
            final_image = np.vstack([combined_image, info_bar])
            
            cv2.imshow("Combined View - Tactile PC + Multi-Camera", cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
        
        episode_id = self.available_episodes[self.current_episode_idx]
        max_frames = len(self.pos_list_0)
        print(f"Episode {episode_id} | Frame {self.current_frame_idx}/{max_frames-1}")
    
    def next_frame(self):
        """Next frame"""
        max_frames = len(self.pos_list_0)
        
        if self.current_frame_idx < max_frames - 1:
            self.current_frame_idx += 1
        else:
            if self.current_episode_idx < len(self.available_episodes) - 1:
                print(f"\nReached last frame of Episode {self.available_episodes[self.current_episode_idx]}, loading next episode...")
                self.current_episode_idx += 1
                self.load_current_episode()
            else:
                print("\nReached the last frame of the last episode!")
    
    def prev_frame(self):
        """Previous frame"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
        else:
            print("Already at first frame!")
    
    def next_episode(self):
        """Next episode"""
        if self.current_episode_idx < len(self.available_episodes) - 1:
            self.current_episode_idx += 1
            print(f"\nSwitched to Episode {self.available_episodes[self.current_episode_idx]}")
            self.load_current_episode()
        else:
            print("Already at last Episode!")
    
    def prev_episode(self):
        """Previous episode"""
        if self.current_episode_idx > 0:
            self.current_episode_idx -= 1
            print(f"\nSwitched to Episode {self.available_episodes[self.current_episode_idx]}")
            self.load_current_episode()
        else:
            print("Already at first Episode!")
    
    def reset_view(self):
        """Reset point cloud view (recalculate fixed camera parameters based on current episode first frame)"""
        self.setup_fixed_camera_params()
        print("📷 Recalculated and fixed point cloud view (based on current episode first frame)")
    
    def record_current_episode(self):
        """Record current episode as video"""
        print(f"\n🎬 Starting to record Episode {self.available_episodes[self.current_episode_idx]}...")
        
        max_frames = len(self.pos_list_0)
        episode_id = self.available_episodes[self.current_episode_idx]
        
        print(f"   Total frames: {max_frames}")
        print(f"   Frame rate: {self.record_fps} fps")
        print(f"   Estimated duration: {max_frames/self.record_fps:.2f} seconds")
        
        self.current_frame_idx = 0
        (visual_1, left_tactile_1, right_tactile_1, visual_0, left_tactile_0, right_tactile_0, 
         left_pc_image_0, right_pc_image_0, left_pc_image_1, right_pc_image_1,
         traj_image_0, traj_image_1) = self.get_current_frame_images()
        
        combined_image = create_bimanual_combined_image(
            visual_1, left_tactile_1, right_tactile_1,
            visual_0, left_tactile_0, right_tactile_0,
            left_pc_image_0, right_pc_image_0, left_pc_image_1, right_pc_image_1,
            traj_image_0, traj_image_1
        )
        
        if combined_image is None:
            print("❌ Cannot generate combined image, recording failed")
            return
        
        img_height, img_width = combined_image.shape[:2]
        info_bar_height = 120
        total_height = img_height + info_bar_height
        total_width = img_width
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            self.output_video, 
            fourcc, 
            self.record_fps, 
            (total_width, total_height)
        )
        
        if not video_writer.isOpened():
            print(f"❌ Cannot create video file: {self.output_video}")
            return
        
        print(f"   Video size: {total_width}x{total_height}")
        print(f"\nStarting recording...")
        
        for frame_idx in range(max_frames):
            self.current_frame_idx = frame_idx
            
            (visual_1, left_tactile_1, right_tactile_1, visual_0, left_tactile_0, right_tactile_0, 
             left_pc_image_0, right_pc_image_0, left_pc_image_1, right_pc_image_1,
             traj_image_0, traj_image_1) = self.get_current_frame_images()
            
            combined_image = create_bimanual_combined_image(
                visual_1, left_tactile_1, right_tactile_1,
                visual_0, left_tactile_0, right_tactile_0,
                left_pc_image_0, right_pc_image_0, left_pc_image_1, right_pc_image_1,
                traj_image_0, traj_image_1
            )
            
            if combined_image is not None:
                info_bar = np.zeros((info_bar_height, img_width, 3), dtype=np.uint8)
                info_bar[:, :] = [30, 30, 30]
                
                info_lines = [
                    f"Episode {episode_id} | Frame {frame_idx}/{max_frames-1} | Time: {frame_idx/self.record_fps:.2f}s",
                ]
                
                current_pos_0 = self.pos_list_0[frame_idx][:3, 3]
                info_lines.append(f"Robot0 Pose: [{current_pos_0[0]:.3f}, {current_pos_0[1]:.3f}, {current_pos_0[2]:.3f}]")
                
                if self.has_robot1_data and frame_idx < len(self.pos_list_1):
                    current_pos_1 = self.pos_list_1[frame_idx][:3, 3]
                    info_lines.append(f"Robot1 Pose: [{current_pos_1[0]:.3f}, {current_pos_1[1]:.3f}, {current_pos_1[2]:.3f}]")
                
                if self.has_gripper0_data and frame_idx < len(self.gripper_widths_0):
                    gripper_width_0 = self.gripper_widths_0[frame_idx]
                    info_lines.append(f"Robot0 Gripper: {gripper_width_0:.4f}m")
                if self.has_gripper1_data and frame_idx < len(self.gripper_widths_1):
                    gripper_width_1 = self.gripper_widths_1[frame_idx]
                    info_lines.append(f"Robot1 Gripper: {gripper_width_1:.4f}m")
                
                pc_info = []
                if self.has_left_tactile_pc_0 and frame_idx < len(self.left_tactile_pointclouds_0):
                    n_points = len(self.left_tactile_pointclouds_0[frame_idx])
                    pc_info.append(f"R0-L:{n_points}")
                if self.has_right_tactile_pc_0 and frame_idx < len(self.right_tactile_pointclouds_0):
                    n_points = len(self.right_tactile_pointclouds_0[frame_idx])
                    pc_info.append(f"R0-R:{n_points}")
                if self.has_left_tactile_pc_1 and frame_idx < len(self.left_tactile_pointclouds_1):
                    n_points = len(self.left_tactile_pointclouds_1[frame_idx])
                    pc_info.append(f"R1-L:{n_points}")
                if self.has_right_tactile_pc_1 and frame_idx < len(self.right_tactile_pointclouds_1):
                    n_points = len(self.right_tactile_pointclouds_1[frame_idx])
                    pc_info.append(f"R1-R:{n_points}")
                if pc_info:
                    info_lines.append(f"Point Clouds: {' | '.join(pc_info)}")
                
                y_offset = 20
                for line in info_lines:
                    cv2.putText(info_bar, line, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    y_offset += 18
                
                final_image = np.vstack([combined_image, info_bar])
                
                frame_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
                
                if (frame_idx + 1) % 10 == 0 or frame_idx == max_frames - 1:
                    progress = (frame_idx + 1) / max_frames * 100
                    print(f"   Progress: {frame_idx + 1}/{max_frames} ({progress:.1f}%)")
        
        video_writer.release()
        
        print(f"\n✅ Recording complete!")
        print(f"   Output file: {self.output_video}")
        print(f"   Total frames: {max_frames}")
        print(f"   Video duration: {max_frames/self.record_fps:.2f} seconds")
        
        self.current_frame_idx = 0
    
    def print_help(self):
        """Print help information"""
        print("\n" + "="*70)
        print("🎮 Combined Visualizer Controls - Tactile PC + Multi-Camera")
        print("="*70)
        print("📱 Keyboard Controls:")
        print("   A/D: Previous/Next Frame (auto-load next episode at last frame)")
        print("   W/S: Previous/Next Episode")
        print("   R: Reset point cloud view")
        print("   Q: Quit")
        print("\n🖼️  Display Layout:")
        print("   Top Row: [R0 Traj] | [R0 Left PC] [R0 Right PC] | [R0 Visual] [R0 L-Tactile] [R0 R-Tactile]")
        print("   Bottom Row: [R1 Traj] | [R1 Left PC] [R1 Right PC] | [R1 Visual] [R1 L-Tactile] [R1 R-Tactile]")
        print("\n🎨 Color Legend:")
        print("   Red trajectory: Robot0 trajectory")
        print("   Green trajectory: Robot1 trajectory")
        print("   Yellow labels: Left tactile point cloud")
        print("   Cyan labels: Right tactile point cloud")
        print("   Red labels: Robot0 cameras")
        print("   Green labels: Robot1 cameras")
        print("\n🎬 Recording Features:")
        print("   --record: Enable recording mode")
        print("   --record_episode N: Record Nth episode (default: 0)")
        print("   --fps N: Recording frame rate (default: 30)")
        print("   --output_video FILE: Specify output file name")
        print("   --continue_after_record: Continue interactive browsing after recording")
        print("\n   Example:")
        print("   python vis_combined_zarr.py --record --record_episode 5 --fps 30")
        print("="*70 + "\n")
    
    def run(self):
        """Run visualization loop"""
        while True:
            self.update_display()
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('d') or key == ord('D'):
                self.next_frame()
            elif key == ord('a') or key == ord('A'):
                self.prev_frame()
            elif key == ord('w') or key == ord('W'):
                self.next_episode()
            elif key == ord('s') or key == ord('S'):
                self.prev_episode()
            elif key == ord('r') or key == ord('R'):
                self.reset_view()
            elif key == ord('q') or key == ord('Q'):
                print("\n👋 Exiting visualizer")
                break
        
        cv2.destroyAllWindows()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Combined visualizer for tactile point clouds and multi-camera images')
    parser.add_argument('zarr_path', nargs='?', 
                       default='/home/drj/codehub/ViTaMIn-B/data/_103_vb_sensor_open_drawer_4/1026_103_vb_sensor_open_drawer_4.zarr.zip',
                       help='Path to zarr.zip file')
    parser.add_argument('--record', type=bool, default=True, 
                       help='Whether to record video')
    parser.add_argument('--record_episode', type=int, default=1,
                       help='Episode index to record (default: 0)')
    parser.add_argument('--output_video', type=str, default=None,
                       help='Output video file path (default: auto-generated)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Recording frame rate (default: 30)')
    parser.add_argument('--continue_after_record', type=bool, default=True,
                       help='Continue interactive browsing after recording')
    
    args = parser.parse_args()
    
    replay_buffer_path = args.zarr_path
    
    if not os.path.exists(replay_buffer_path):
        print(f"❌ Cannot find zarr file: {replay_buffer_path}")
        print("Please check the file path or specify on command line:")
        print(f"python {sys.argv[0]} <zarr_file_path>")
        return
    
    print(f"🔍 Loading: {replay_buffer_path}")
    
    with zarr.ZipStore(replay_buffer_path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(src_store=zip_store, store=zarr.MemoryStore())
    
    print(f"✅ Successfully loaded ReplayBuffer")
    print(f"   Total frames: {replay_buffer.n_steps}")
    print(f"   Number of episodes: {replay_buffer.n_episodes}")
    
    available_episodes = np.arange(replay_buffer.n_episodes)
    
    if args.record:
        if args.record_episode >= replay_buffer.n_episodes:
            print(f"❌ Episode {args.record_episode} out of range ({replay_buffer.n_episodes} episodes total)")
            return
        
        print(f"\n🎬 Recording mode starting...")
        print(f"   Recording Episode: {args.record_episode}")
        print(f"   Frame rate: {args.fps} fps")
        
        if args.output_video is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            zarr_name = os.path.basename(replay_buffer_path).replace('.zarr.zip', '')
            args.output_video = f"recorded_ep{args.record_episode}_{zarr_name}_{timestamp}.mp4"
        
        print(f"   Output file: {args.output_video}")
    
    print(f"\n🚀 Launching combined visualizer...")
    
    try:
        visualizer = CombinedVisualizer(
            replay_buffer, 
            available_episodes,
            record_mode=args.record,
            record_episode=args.record_episode,
            output_video=args.output_video,
            record_fps=args.fps,
            continue_after_record=args.continue_after_record
        )
    except Exception as e:
        print(f"❌ Visualizer startup failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

