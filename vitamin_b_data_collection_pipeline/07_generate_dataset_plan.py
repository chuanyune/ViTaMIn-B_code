import sys
import os
import argparse
from omegaconf import OmegaConf
from pathlib import Path

project_root = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(project_root))

import pathlib
import pickle
import numpy as np
import json
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import av
from utils.pose_util import mat_to_pose
from datetime import datetime

def find_video_files(demo_dir, mode=None, use_tactile=False, available_hands=None):
    """Find all video files in the demo directory based on mode and available hands"""
    videos_dir = demo_dir.joinpath('videos')
    video_files = {}
    
    if not videos_dir.exists():
        return video_files
    
    if mode == 'single':
        # Single hand mode: detect which hand is actually being used
        # Check for visual videos to determine the actual hand
        left_visual_path = videos_dir.joinpath('left_hand_visual.mp4')
        right_visual_path = videos_dir.joinpath('right_hand_visual.mp4')
        
        actual_hand = None
        if left_visual_path.exists() and not right_visual_path.exists():
            actual_hand = 'left'
            video_files['left_visual'] = left_visual_path
        elif right_visual_path.exists() and not left_visual_path.exists():
            actual_hand = 'right'
            video_files['right_visual'] = right_visual_path
        elif available_hands and len(available_hands) == 1:
            # Fallback to available_hands if visual detection is ambiguous
            actual_hand = available_hands[0]
            if actual_hand == 'left' and left_visual_path.exists():
                video_files['left_visual'] = left_visual_path
            elif actual_hand == 'right' and right_visual_path.exists():
                video_files['right_visual'] = right_visual_path
        
        # Add tactile videos if enabled, using the correct hand mapping
        if use_tactile and actual_hand:
            if actual_hand == 'left':
                # Left hand: use left_hand tactile sensors
                tactile_left = videos_dir.joinpath('left_hand_left_tactile.mp4')
                tactile_right = videos_dir.joinpath('left_hand_right_tactile.mp4')
                if tactile_left.exists():
                    video_files['left_hand_left_tactile'] = tactile_left
                if tactile_right.exists():
                    video_files['left_hand_right_tactile'] = tactile_right
            elif actual_hand == 'right':
                # Right hand: use right_hand tactile sensors
                tactile_left = videos_dir.joinpath('right_hand_left_tactile.mp4')
                tactile_right = videos_dir.joinpath('right_hand_right_tactile.mp4')
                if tactile_left.exists():
                    video_files['right_hand_left_tactile'] = tactile_left
                if tactile_right.exists():
                    video_files['right_hand_right_tactile'] = tactile_right
    else:
        # Bimanual mode: look for left_hand_visual.mp4 and right_hand_visual.mp4
        left_visual_path = videos_dir.joinpath('left_hand_visual.mp4')
        right_visual_path = videos_dir.joinpath('right_hand_visual.mp4')
        
        if left_visual_path.exists():
            video_files['left_visual'] = left_visual_path
        if right_visual_path.exists():
            video_files['right_visual'] = right_visual_path
            
        # Add tactile videos if enabled
        if use_tactile:
            # Left hand tactile
            left_tactile_left = videos_dir.joinpath('left_hand_left_tactile.mp4')
            left_tactile_right = videos_dir.joinpath('left_hand_right_tactile.mp4')
            if left_tactile_left.exists():
                video_files['left_hand_left_tactile'] = left_tactile_left
            if left_tactile_right.exists():
                video_files['left_hand_right_tactile'] = left_tactile_right
                
            # Right hand tactile
            right_tactile_left = videos_dir.joinpath('right_hand_left_tactile.mp4')
            right_tactile_right = videos_dir.joinpath('right_hand_right_tactile.mp4')
            if right_tactile_left.exists():
                video_files['right_hand_left_tactile'] = right_tactile_left
            if right_tactile_right.exists():
                video_files['right_hand_right_tactile'] = right_tactile_right
            
    return video_files

def detect_demo_mode(demo_dir):
    """
    Detect if demo is single-hand or bimanual mode based on trajectory files and video files.
    
    Returns:
        tuple: (mode, available_hands) where mode is 'single', 'bimanual', or None
    """
    pose_dir = demo_dir.joinpath('pose_data')
    videos_dir = demo_dir.joinpath('videos')
    
    # First check based on video files
    # After pipeline updates, we now use left_hand_visual.mp4 and right_hand_visual.mp4 in videos/ directory
    left_visual_exists = videos_dir.joinpath('left_hand_visual.mp4').exists()
    right_visual_exists = videos_dir.joinpath('right_hand_visual.mp4').exists()
    
    # Check if we have both left and right hand visuals (bimanual mode)
    if right_visual_exists and left_visual_exists:
        # Bimanual mode: both hands present
        return "bimanual", ['left', 'right']
    elif left_visual_exists and not right_visual_exists:
        # Single hand mode with left visual: check which hands have complete data
        available_hands = []
        
        if pose_dir.exists():
            # Check left hand
            left_traj_exists = pose_dir.joinpath('left_hand_trajectory_synced.csv').exists()
            left_gripper_exists = demo_dir.joinpath('gripper_width_left.csv').exists()
            if left_traj_exists and left_gripper_exists:
                available_hands.append('left')
            
            # Check right hand (in case left visual is used but right hand data exists)
            right_traj_exists = pose_dir.joinpath('right_hand_trajectory_synced.csv').exists()
            right_gripper_exists = demo_dir.joinpath('gripper_width_right.csv').exists()
            if right_traj_exists and right_gripper_exists:
                available_hands.append('right')
            
            if available_hands:
                return "single", available_hands
            else:
                return None, []
        
        # Default fallback: assume left hand if no pose data exists
        return "single", ['left']
    elif right_visual_exists and not left_visual_exists:
        # Single hand mode with right visual: check which hands have complete data
        available_hands = []
        
        if pose_dir.exists():
            # Check right hand
            right_traj_exists = pose_dir.joinpath('right_hand_trajectory_synced.csv').exists()
            right_gripper_exists = demo_dir.joinpath('gripper_width_right.csv').exists()
            if right_traj_exists and right_gripper_exists:
                available_hands.append('right')
            
            # Check left hand (in case right visual is used but left hand data exists)
            left_traj_exists = pose_dir.joinpath('left_hand_trajectory_synced.csv').exists()
            left_gripper_exists = demo_dir.joinpath('gripper_width_left.csv').exists()
            if left_traj_exists and left_gripper_exists:
                available_hands.append('left')
            
            if available_hands:
                return "single", available_hands
            else:
                return None, []
        
        # Default fallback: assume right hand if no pose data exists
        return "single", ['right']
    
    # Fallback: check based on trajectory files only
    if pose_dir.exists():
        available_hands = []
        
        # Check left hand (both trajectory and gripper data must exist)
        left_traj_exists = pose_dir.joinpath('left_hand_trajectory_synced.csv').exists()
        left_gripper_exists = demo_dir.joinpath('gripper_width_left.csv').exists()
        if left_traj_exists and left_gripper_exists:
            available_hands.append('left')
        
        # Check right hand (both trajectory and gripper data must exist)
        right_traj_exists = pose_dir.joinpath('right_hand_trajectory_synced.csv').exists()
        right_gripper_exists = demo_dir.joinpath('gripper_width_right.csv').exists()
        if right_traj_exists and right_gripper_exists:
            available_hands.append('right')
        
        if len(available_hands) >= 2:
            return "bimanual", available_hands
        elif len(available_hands) == 1:
            return "single", available_hands
    
    return None, []

def check_aruco_detection_files(demo_dir, mode, available_hands):
    """
    Check if required ArUco detection files exist based on mode and available hands.
    
    Returns:
        bool: True if all required files exist
    """
    if mode == 'single':
        # Single hand mode: check for tag_detection_left.pkl
        aruco_pkl_path = demo_dir.joinpath('tag_detection_left.pkl')
        return aruco_pkl_path.exists()
    
    elif mode == 'bimanual':
        # Bimanual mode: check for hand-specific pkl files
        missing_files = []
        
        if 'right' in available_hands:
            right_pkl_path = demo_dir.joinpath('tag_detection_right.pkl')
            if not right_pkl_path.exists():
                missing_files.append('tag_detection_right.pkl')
        
        if 'left' in available_hands:
            left_pkl_path = demo_dir.joinpath('tag_detection_left.pkl')
            if not left_pkl_path.exists():
                missing_files.append('tag_detection_left.pkl')
        
        if missing_files:
            print(f"   [ERROR] Missing ArUco files: {', '.join(missing_files)}")
            return False
        
        return True
    
    return False

def process_hand_trajectory(demo_dir, position, n_frames, mode='single'):
    """Process trajectory data for a specific hand"""
    pose_dir = demo_dir.joinpath('pose_data')
    traj_path = pose_dir.joinpath(f"{position}_hand_trajectory_synced.csv")
    
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    
    traj_df = pd.read_csv(traj_path)
    
    # Load hand pose data
    df = traj_df.iloc[:n_frames]
    _quest_pos = df[['x', 'y', 'z']].to_numpy()
    _quest_rot_quat_xyzw = df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy()
    
    _quest_rot = Rotation.from_quat(_quest_rot_quat_xyzw)
    _quest_pose = np.zeros((_quest_pos.shape[0], 4, 4), dtype=np.float32)
    _quest_pose[:,3,3] = 1
    _quest_pose[:,:3,3] = _quest_pos
    _quest_pose[:,:3,:3] = _quest_rot.as_matrix()

    # Load gripper width data based on mode
    gripper_path = demo_dir.joinpath(f'gripper_width_{position}.csv')
    
    if not gripper_path.exists():
        raise FileNotFoundError(f"Gripper width file not found: {gripper_path}")
    
    gripper_data = pd.read_csv(gripper_path)
    # Check column names, support two formats: 'width' or 'gripper_width'
    if 'width' in gripper_data.columns:
        gripper_widths = gripper_data['width'].values[:n_frames]
    elif 'gripper_width' in gripper_data.columns:
        gripper_widths = gripper_data['gripper_width'].values[:n_frames]
    else:
        raise ValueError(f"Gripper width file {gripper_path} missing 'width' or 'gripper_width' column")

    quest_pose = mat_to_pose(_quest_pose)
    
    return {
        "quest_pose": quest_pose,
        "gripper_width": gripper_widths,
        "demo_start_pose": quest_pose[0],
        "demo_end_pose": quest_pose[-1]
    }

def create_camera_entries(video_files, demo_dir, n_frames, mode="single", hands=None):
    """Create camera entries for dataset plan"""
    cameras = []
    
    if mode == "single":
        # Single hand mode: simple camera entries
        for usage_name, video_path in video_files.items():
            # demo_dir is data/_817/demos/demo_xxx, we want relative to data/_817/demos
            camera_entry = {
                "video_path": str(video_path.relative_to(demo_dir.parent)),
                "video_start_end": (0, n_frames),
                "usage_name": usage_name,
                "hand_position_idx": 0  # Single hand always gets index 0
            }
            cameras.append(camera_entry)
    
    elif mode == "bimanual":
        # Bimanual mode: assign hand_position_idx based on hand
        hand_idx_map = {'left': 0, 'right': 1}
        
        for usage_name, video_path in video_files.items():
            # Determine which hand this camera belongs to
            camera_hand = None
            for hand in hands:
                if hand in usage_name:
                    camera_hand = hand
                    break
            
            if camera_hand is None:
                print(f"Warning: Cannot determine hand for camera {usage_name}, skipping")
                continue
            
            # demo_dir is data/_817/demos/demo_xxx, we want relative to data/_817/demos
            camera_entry = {
                "video_path": str(video_path.relative_to(demo_dir.parent)),
                "video_start_end": (0, n_frames),
                "usage_name": usage_name,
                "position": camera_hand,
                "hand_position_idx": hand_idx_map[camera_hand]
            }
            cameras.append(camera_entry)
    
    return cameras

def get_demo_metadata(demo_dir):
    """Get demo metadata (timestamps, fps, etc.)"""
    pose_dir = demo_dir.joinpath('pose_data')
    
    # Try to find metadata files
    for meta_file in ['left_metadata.json', 'right_metadata.json']:
        meta_path = pose_dir.joinpath(meta_file)
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                return json.load(f)
    
    return None

def process_single_demo(demo_dir, min_episode_length, fps=None, use_tactile=False):
    """
    Process a single demo directory and return dataset plan entry.
    
    Args:
        demo_dir: Path to demo directory
        min_episode_length: Minimum required episode length
        fps: Expected FPS (for validation)
        use_tactile: Whether to process tactile videos
    
    Returns:
        dict: Dataset plan entry or None if processing failed
    """
    try:
        print(f"\n Processing demo: {demo_dir.name}")
        
        # Detect demo mode
        demo_mode, available_hands = detect_demo_mode(demo_dir)
        if demo_mode is None:
            print(f"   [ERROR] No trajectory data found, skipping")
            return None
        
        print(f"   Mode: {demo_mode}, Hands: {available_hands}")
        print(f"    Tactile data: {'enabled' if use_tactile else 'disabled'}")
        
        # Check for ArUco detection results based on mode
        if not check_aruco_detection_files(demo_dir, demo_mode, available_hands):
            print(f"   [ERROR] Missing ArUco detection results (run 07_A-detect_aruco.py first), skipping")
            return None
        
        # Find video files based on mode and available hands
        video_files = find_video_files(demo_dir, demo_mode, use_tactile, available_hands)
        if not video_files:
            print(f"   [ERROR] No video files found, skipping")
            return None
        
        # Get reference video for frame count and FPS
        main_video_key = None
        if demo_mode == 'single':
            # Single hand mode: use left_visual (left_hand_visual.mp4)
            if 'left_visual' in video_files:
                main_video_key = 'left_visual'
        else:
            # Bimanual mode: prefer left_visual, then right_visual
            if 'left_visual' in video_files:
                main_video_key = 'left_visual'
            elif 'right_visual' in video_files:
                main_video_key = 'right_visual'
        
        # Fallback to any available video
        if main_video_key is None:
            for usage_name in video_files.keys():
                if "visual" in usage_name:
                    main_video_key = usage_name
                    break
        
        if main_video_key is None:
            main_video_key = list(video_files.keys())[0]
        
        main_video_path = video_files[main_video_key]
        
        # Get video information
        with av.open(str(main_video_path), 'r') as container:
            stream = container.streams.video[0]
            n_frames = stream.frames
            video_fps = float(stream.average_rate)

        
        # Validate FPS consistency
        if fps is not None and abs(fps - video_fps) > 0.01:
            print(f"   [WARNING]  Inconsistent FPS: expected {fps}, got {video_fps}")
            return None
        
        # Check minimum episode length
        if n_frames < min_episode_length:
            print(f"   [ERROR] Episode too short: {n_frames} < {min_episode_length}")
            return None
        
        # Validate data consistency across hands
        min_frames = n_frames
        for hand in available_hands:
            # Check trajectory data
            traj_path = demo_dir.joinpath('pose_data', f"{hand}_hand_trajectory_synced.csv")
            if traj_path.exists():
                traj_df = pd.read_csv(traj_path)
                min_frames = min(min_frames, len(traj_df))
                print(f"   [INFO] {hand.capitalize()} trajectory: {len(traj_df)} points")
            
            # Check gripper data based on mode
            gripper_path = demo_dir.joinpath(f'gripper_width_{hand}.csv')
            
            if gripper_path.exists():
                gripper_data = pd.read_csv(gripper_path)
                min_frames = min(min_frames, len(gripper_data))
                print(f"   [INFO] {hand.capitalize()} gripper: {len(gripper_data)} points")
        
        if min_frames != n_frames:
            print(f"   [WARNING]  Data length mismatch: video {n_frames}, min data {min_frames}")
            n_frames = min_frames
        
        # Process hand trajectories
        grippers = []
        for hand in available_hands:
            try:
                gripper_data = process_hand_trajectory(demo_dir, hand, n_frames, demo_mode)
                grippers.append(gripper_data)
                print(f"   [SUCCESS] {hand.capitalize()} hand processed: {n_frames} frames")
            except Exception as e:
                print(f"   [ERROR] Failed to process {hand} hand: {e}")
                return None
        
        # Create camera entries
        cameras = create_camera_entries(video_files, demo_dir, n_frames, demo_mode, available_hands)
        
        # Get demo metadata for timestamps
        demo_meta = get_demo_metadata(demo_dir)
        if demo_meta and 'start_time' in demo_meta:
            start_time = demo_meta['start_time']
        else:
            # Fallback: extract from demo directory name or use current time
            try:
                if 'demo_' in demo_dir.name:
                    datetime_part = demo_dir.name.split('_', 2)[-1]
                    start_date = datetime.strptime(datetime_part, '%Y.%m.%d_%H.%M.%S.%f')
                    start_time = start_date.timestamp()
                else:
                    start_time = datetime.now().timestamp()
            except:
                start_time = datetime.now().timestamp()
        
        # Create episode timestamps
        dt = 1.0 / video_fps
        episode_timestamps = np.arange(n_frames) * dt + start_time
        
        print(f"   [SUCCESS] Success: {len(grippers)} gripper(s), {len(cameras)} camera(s), {n_frames} frames")
        
        return {
            "episode_timestamps": episode_timestamps,
            "grippers": grippers,
            "cameras": cameras,
            "demo_mode": demo_mode,
            "demo_name": demo_dir.name,
            "n_frames": n_frames,
            "fps": video_fps
        }
        
    except Exception as e:
        print(f"   [ERROR] Processing failed: {e}")
        return None

def main(input_path: str, min_episode_length: int, use_tactile: bool):
    """Main processing function"""
    input_path = pathlib.Path(os.path.expanduser(input_path)).absolute()
    demos_dir = Path(os.path.join(input_path, 'demos'))
    output = input_path.joinpath('dataset_plan.pkl')

    print(f"Processing dataset: {demos_dir}")
    print(f"[INFO] Minimum episode length: {min_episode_length}")
    print(f" Tactile data: {'enabled' if use_tactile else 'disabled'}")

    # Find all demo directories
    demo_dirs = sorted([x for x in demos_dir.glob('demo_*') if x.is_dir()])
    
    if not demo_dirs:
        print("[ERROR] No demo directories found!")
        return
    
    print(f" Found {len(demo_dirs)} demo directories")

    # Process demos
    all_plans = []
    fps = None
    stats = {
        'total_demos': len(demo_dirs),
        'processed_demos': 0,
        'skipped_demos': 0,
        'single_hand_demos': 0,
        'bimanual_demos': 0,
        'total_frames': 0,
        'total_duration': 0.0
    }

    for demo_dir in demo_dirs:
        plan = process_single_demo(demo_dir, min_episode_length, fps, use_tactile)
        
        if plan is not None:
            all_plans.append(plan)
            stats['processed_demos'] += 1
            stats['total_frames'] += plan['n_frames']
            stats['total_duration'] += plan['n_frames'] / plan['fps']
            
            # Set reference FPS from first valid demo
            if fps is None:
                fps = plan['fps']
                print(f"Reference FPS: {fps}")
            
            # Count demo types
            if plan['demo_mode'] == 'single':
                stats['single_hand_demos'] += 1
            elif plan['demo_mode'] == 'bimanual':
                stats['bimanual_demos'] += 1
        else:
            stats['skipped_demos'] += 1

    # Print final statistics
    print(f"\n{'='*60}")
    print(f"[INFO] PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total demos found: {stats['total_demos']}")
    print(f"Successfully processed: {stats['processed_demos']}")
    print(f"Skipped demos: {stats['skipped_demos']}")
    print(f"Single-hand demos: {stats['single_hand_demos']}")
    print(f"Bimanual demos: {stats['bimanual_demos']}")
    print(f"Total frames: {stats['total_frames']:,}")
    print(f"Total duration: {stats['total_duration']:.1f} seconds ({stats['total_duration']/60:.1f} minutes)")
    
    if stats['processed_demos'] > 0:
        success_rate = stats['processed_demos'] / stats['total_demos'] * 100
        avg_episode_length = stats['total_frames'] / stats['processed_demos']
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average episode length: {avg_episode_length:.1f} frames")
    
    if len(all_plans) == 0:
        print("[ERROR] No valid demos processed!")
        return

    # Save dataset plan
    print(f"\nSaving dataset plan to: {output}")
    pickle.dump(all_plans, output.open('wb'))
    print("[SUCCESS] Dataset plan saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dataset plan from recorded demonstrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 09_generate_dataset_plan.py --cfg config/data_collection.yaml
        """)
    parser.add_argument('--cfg', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    config_file = args.cfg
    cfg = OmegaConf.load(config_file)
    
    task_name = cfg["task"]["name"]
    min_episode_length = cfg["output_train_data"]["min_episode_length"]
    # If tactile image or point cloud processing is enabled, tactile videos need to be included
    use_tactile_img = cfg["output_train_data"].get("use_tactile_img", False)
    use_tactile_pc = cfg["output_train_data"].get("use_tactile_pc", False)
    use_tactile = use_tactile_img or use_tactile_pc
    input_path = os.path.join("data", task_name)

    print(f"Task name: {task_name}")

    main(input_path, min_episode_length, use_tactile)