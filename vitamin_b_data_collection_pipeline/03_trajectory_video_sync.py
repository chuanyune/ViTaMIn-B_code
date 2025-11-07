#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import csv
import subprocess
import concurrent.futures
import multiprocessing
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from omegaconf import OmegaConf
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Global processing configuration
PROCESSING_CONFIG = {
    'encoder': 'libx264',
    'preset': 'fast',
    'max_parallel_jobs': 4
}

# Global logging configuration
logger = logging.getLogger(__name__)

def setup_logging(task_dir: Path) -> Path:
    """
    Setup logging system and create detailed log file
    
    Args:
        task_dir: Task directory
        
    Returns:
        Path: Log file path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = task_dir / f"trajectory_sync_detailed_{timestamp}.log"
    
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    
    logger.info("="*80)
    logger.info("TRAJECTORY VIDEO SYNCHRONIZATION - DETAILED LOG")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    return log_file

def log_and_print(message: str, level: str = "info", print_to_console: bool = False):
    """
    Log message and optionally print to console
    
    Args:
        message: Log message
        level: Log level (debug, info, warning, error)
        print_to_console: Whether to print to console
    """
    if level == "debug":
        logger.debug(message)
    elif level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    
    if print_to_console:
        print(message)


def configure_parallel_processing(max_parallel_jobs: int = None) -> None:
    """
    Configure CPU parallel processing settings
    
    Args:
        max_parallel_jobs: Maximum parallel jobs, None for auto-detect
    """
    global PROCESSING_CONFIG
    
    cpu_count = multiprocessing.cpu_count()
    if max_parallel_jobs is None:
        max_parallel_jobs = min(cpu_count, 6)
    
    PROCESSING_CONFIG.update({
        'encoder': 'libx264',
        'preset': 'fast',
        'max_parallel_jobs': max_parallel_jobs
    })
    
    print(f"\nCPU PARALLEL PROCESSING CONFIG")
    print(f"Processing mode: CPU Only")
    print(f"Max parallel jobs: {PROCESSING_CONFIG['max_parallel_jobs']}")
    print(f"CPU cores available: {cpu_count}")
    
    logger.info("CPU PARALLEL PROCESSING CONFIG")
    logger.info(f"Encoder: {PROCESSING_CONFIG['encoder']}")
    logger.info(f"Preset: {PROCESSING_CONFIG['preset']}")
    logger.info(f"Max parallel jobs: {PROCESSING_CONFIG['max_parallel_jobs']}")
    logger.info(f"CPU cores: {cpu_count}")

def load_trajectory_json_files(trajectory_dir: Path) -> List[Dict]:
    """Load Quest trajectory JSON files."""
    log_and_print(f"Loading trajectory from: {trajectory_dir}", "info", True)
    logger.info(f"Loading trajectory from directory: {trajectory_dir}")
    
    # Find all Quest pose JSON files
    json_files = list(trajectory_dir.glob("quest_poses_*.json"))
    if not json_files:
        error_msg = f"No Quest pose JSON files found in {trajectory_dir}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Sort files by name to ensure correct order
    json_files.sort()
    log_and_print(f"Found {len(json_files)} JSON files", "info", True)
    logger.info(f"Found JSON files: {[f.name for f in json_files]}")
    
    all_data = []
    
    for json_file in tqdm(json_files, desc="Loading trajectory files", disable=False):
        logger.debug(f"Loading file: {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        logger.debug(f"File {json_file.name} contains {len(file_data)} frames")
        
        # Convert Quest JSON format to trajectory format
        for frame in file_data:
            # Extract left wrist data
            left_wrist = frame['left_wrist']
            right_wrist = frame['right_wrist']
            
            row = {
                'timestamp': frame['timestamp_unix'],
                'left_wrist_x': left_wrist['position']['x'],
                'left_wrist_y': left_wrist['position']['y'], 
                'left_wrist_z': left_wrist['position']['z'],
                'left_wrist_qx': left_wrist['rotation']['x'],
                'left_wrist_qy': left_wrist['rotation']['y'],
                'left_wrist_qz': left_wrist['rotation']['z'],
                'left_wrist_qw': left_wrist['rotation']['w'],
                'right_wrist_x': right_wrist['position']['x'],
                'right_wrist_y': right_wrist['position']['y'],
                'right_wrist_z': right_wrist['position']['z'],
                'right_wrist_qx': right_wrist['rotation']['x'],
                'right_wrist_qy': right_wrist['rotation']['y'],
                'right_wrist_qz': right_wrist['rotation']['z'],
                'right_wrist_qw': right_wrist['rotation']['w']
            }
            all_data.append(row)
    
    all_data.sort(key=lambda x: x['timestamp'])
    
    result_msg = f"[SUCCESS] Loaded {len(all_data)} trajectory points from {len(json_files)} files"
    log_and_print(result_msg, "info", True)
    logger.info(f"Total trajectory points loaded: {len(all_data)}")
    logger.info(f"Trajectory time range: {all_data[0]['timestamp']:.6f} - {all_data[-1]['timestamp']:.6f}")
    
    return all_data

def load_video_timestamps(file_path: Path) -> List[Dict]:
    """Load video timestamps CSV file."""
    print(f"Loading video timestamps from: {file_path.name}")
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Convert timestamps to float
    for row in data:
        row['timestamp'] = float(row['timestamp'])
    
    print(f"[SUCCESS] Loaded {len(data)} video timestamps")
    return data

def get_default_world_frame() -> np.ndarray:
    """Get default world frame: origin with identity rotation."""
    # Default world frame: [x, y, z, qx, qy, qz, qw] = [0, 0, 0, 0, 0, 0, 1]
    world_frame = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    print(f"Using default world frame: {world_frame}")
    return world_frame

def compute_rel_transform(pose: np.ndarray, world_frame: np.ndarray) -> tuple:
    """
    Compute relative transform from unity pose to world frame.
    
    Args:
        pose: np.ndarray shape (7,) [x, y, z, qx, qy, qz, qw] in unity frame
        world_frame: np.ndarray shape (7,) [x, y, z, qx, qy, qz, qw] world frame reference
        
    Returns:
        tuple: (rel_pos, rel_quat) relative position and rotation
    """
    world_frame = world_frame.copy()
    pose = pose.copy()
    
    # Convert unity coordinates: [x, y, z] -> [x, z, y]
    world_frame[:3] = np.array([world_frame[0], world_frame[2], world_frame[1]])
    pose[:3] = np.array([pose[0], pose[2], pose[1]])

    # Coordinate transformation matrix
    Q = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0.]])
    
    # Compute relative rotation
    rot_base = Rotation.from_quat(world_frame[3:]).as_matrix()
    rot = Rotation.from_quat(pose[3:]).as_matrix()
    rel_rot = Rotation.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    
    # Compute relative position
    rel_pos = Rotation.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
    
    return rel_pos, rel_rot.as_quat()

def crop_trajectory_by_time(trajectory_data: List[Dict], 
                          start_time: float, end_time: float) -> List[Dict]:
    """
    Crop trajectory data based on time range.
    
    Args:
        trajectory_data: Full trajectory data
        start_time: Video start time
        end_time: Video end time
        
    Returns:
        Cropped trajectory data
    """
    print(f"Cropping trajectory:")
    print(f"   Time range: {start_time:.3f} - {end_time:.3f}")
    
    cropped_data = []
    for row in trajectory_data:
        if start_time <= row['timestamp'] <= end_time:
            cropped_data.append(row)
    
    print(f"Cropped to {len(cropped_data)} points")
    return cropped_data

def interpolate_timestamps(start_time: float, end_time: float, count: int) -> List[float]:
    """Generate evenly spaced timestamps."""
    if count == 1:
        return [start_time]
    
    step = (end_time - start_time) / (count - 1)
    return [start_time + i * step for i in range(count)]

def simple_interpolate(timestamps: List[float], values: List[float], 
                      target_timestamps: List[float]) -> List[float]:
    """Simple linear interpolation."""
    result = []
    
    for target_t in target_timestamps:
        # Find surrounding points
        if target_t <= timestamps[0]:
            result.append(values[0])
        elif target_t >= timestamps[-1]:
            result.append(values[-1])
        else:
            # Find the interval
            for i in range(len(timestamps) - 1):
                if timestamps[i] <= target_t <= timestamps[i + 1]:
                    # Linear interpolation
                    t0, t1 = timestamps[i], timestamps[i + 1]
                    v0, v1 = values[i], values[i + 1]
                    
                    if t1 == t0:
                        interpolated = v0
                    else:
                        ratio = (target_t - t0) / (t1 - t0)
                        interpolated = v0 + ratio * (v1 - v0)
                    
                    result.append(interpolated)
                    break
    
    return result

def generate_hand_trajectory(cropped_data: List[Dict], hand: str,
                           video_start: float, video_end: float, 
                           frame_count: int,
                           world_frame: np.ndarray) -> List[Dict]:
    """
    Generate synchronized hand trajectory with world frame transformation.
    
    Args:
        cropped_data: Cropped trajectory data
        hand: 'left' or 'right'
        video_start: Video start timestamp
        video_end: Video end timestamp
        frame_count: Number of video frames
        world_frame: World frame reference for coordinate transformation
        
    Returns:
        Synchronized hand trajectory data with relative coordinates
    """
    # Swap left and right hands as requested
    source_hand = 'right_wrist' if hand == 'left' else 'left_wrist'
    print(f"Generating {hand} hand using {source_hand} data (swapped)")
    
    # Create target timestamps (video timeline)
    target_timestamps = interpolate_timestamps(video_start, video_end, frame_count)
    
    # Extract trajectory timestamps
    traj_timestamps = [row['timestamp'] for row in cropped_data]
    
    # Extract position and rotation data
    pos_cols = ['x', 'y', 'z']
    quat_cols = ['qx', 'qy', 'qz', 'qw']
    
    synchronized_data = []
    
    # Extract all positions and quaternions for vectorized interpolation
    positions = []
    quaternions = []
    
    for row in cropped_data:
        pos = [float(row[f'{source_hand}_{col}']) for col in pos_cols]
        quat = [float(row[f'{source_hand}_{col}']) for col in quat_cols]
        positions.append(pos)
        quaternions.append(quat)
    
    positions = np.array(positions)  # Shape: (n_points, 3)
    quaternions = np.array(quaternions)  # Shape: (n_points, 4)
    
    # Check minimum data requirements
    if len(cropped_data) < 2:
        print(f"Warning: Only {len(cropped_data)} trajectory points available, using simple duplication")
        # For single point or no data, duplicate the available data
        if len(cropped_data) == 1:
            interpolated_positions = np.tile(positions[0], (len(target_timestamps), 1))
            interpolated_quaternions = np.tile(quaternions[0], (len(target_timestamps), 1))
        else:
            # No data case - this should have been caught earlier, but handle gracefully
            raise ValueError("No trajectory data available for interpolation")
    else:
        # Normalize quaternions for SLERP
        quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
        
        # Check for degenerate quaternions (near-zero norm)
        quat_norms = np.linalg.norm(quaternions, axis=1)
        if np.any(quat_norms < 1e-6):
            print(f"Warning: Found degenerate quaternions, attempting to fix...")
            # Replace degenerate quaternions with identity quaternion
            degenerate_mask = quat_norms < 1e-6
            quaternions[degenerate_mask] = [0, 0, 0, 1]
        
        # Create Rotation objects for SLERP
        try:
            rotations = Rotation.from_quat(quaternions)
        except Exception as e:
            print(f"Warning: Invalid quaternions detected, using linear interpolation fallback: {e}")
            # Fallback to linear interpolation for quaternions
            from scipy.interpolate import interp1d
            position_interpolator = interp1d(traj_timestamps, positions.T, 
                                           kind='linear', axis=1, fill_value='extrapolate')
            quat_interpolator = interp1d(traj_timestamps, quaternions.T, 
                                       kind='linear', axis=1, fill_value='extrapolate')
            
            interpolated_positions = position_interpolator(target_timestamps).T
            interpolated_quaternions = quat_interpolator(target_timestamps).T
            # Normalize interpolated quaternions
            interpolated_quaternions = interpolated_quaternions / np.linalg.norm(interpolated_quaternions, axis=1, keepdims=True)
        else:
            # Normal SLERP interpolation
            # Clamp target timestamps to trajectory range to avoid extrapolation
            traj_start, traj_end = traj_timestamps[0], traj_timestamps[-1]
            clamped_target_timestamps = np.clip(target_timestamps, traj_start, traj_end)
            
            # Position interpolation (linear)
            from scipy.interpolate import interp1d
            position_interpolator = interp1d(traj_timestamps, positions.T, 
                                           kind='linear', axis=1, 
                                           fill_value='extrapolate')
            interpolated_positions = position_interpolator(clamped_target_timestamps).T
            
            # Rotation interpolation (SLERP)
            from scipy.spatial.transform import Slerp
            slerp = Slerp(traj_timestamps, rotations)
            interpolated_rotations = slerp(clamped_target_timestamps)
            interpolated_quaternions = interpolated_rotations.as_quat()
            
            print(f"Using SLERP interpolation for smooth rotation")
    
    # Generate synchronized data points
    for i, target_t in enumerate(target_timestamps):
        # Get interpolated position and quaternion
        pos = interpolated_positions[i]
        quat = interpolated_quaternions[i]
        
        # Create pose vector [x, y, z, qx, qy, qz, qw]
        pose = np.concatenate([pos, quat])
        
        # Apply world frame transformation
        rel_pos, rel_quat = compute_rel_transform(pose, world_frame)
        
        # Create final point with relative coordinates
        point = {
            'timestamp': target_t,
            'x': rel_pos[0],
            'y': rel_pos[1], 
            'z': rel_pos[2],
            'q_x': rel_quat[0],
            'q_y': rel_quat[1],
            'q_z': rel_quat[2],
            'q_w': rel_quat[3]
        }
        synchronized_data.append(point)
    
    print(f"[SUCCESS] Generated {len(synchronized_data)} synchronized points")
    return synchronized_data

def save_hand_trajectory(data: List[Dict], output_file: Path):
    """Save hand trajectory to CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w'])
        
        for row in data:
            writer.writerow([
                row['timestamp'],
                row['x'], row['y'], row['z'],
                row['q_x'], row['q_y'], row['q_z'], row['q_w']
            ])
    
    print(f"Saved: {output_file}")





def process_demo(demo_dir: Path, trajectory_data: List[Dict], world_frame: np.ndarray, config: OmegaConf) -> bool:
    """Process a single demo directory."""
    try:
        print(f"\n{'='*80}")
        print(f"PROCESSING DEMO: {demo_dir.name}")
        print(f"{'='*80}")
        
        # Find video timestamps
        videos_dir = demo_dir / "videos"
        if not videos_dir.exists():
            print(f"[ERROR] Videos directory not found: {videos_dir}")
            return False
        
        timestamp_files = list(videos_dir.glob("*_timestamps.csv"))
        if not timestamp_files:
            print(f"[ERROR] No timestamp files found in {videos_dir}")
            return False
        
        # Analyze video files to determine mode
        visual_timestamp_files = [f for f in timestamp_files if 'visual' in f.name]
        tactile_timestamp_files = [f for f in timestamp_files if 'tactile' in f.name]
        
        demo_mode = detect_demo_mode_from_config(config)
        print(f"Demo mode (from config): {demo_mode}")
        
        verify_config_file_consistency(demo_mode, visual_timestamp_files, tactile_timestamp_files)
        
        if tactile_timestamp_files:
            print(f"Detected tactile videos - performing global video synchronization")
            tactile_sync_success = process_tactile_videos(demo_dir, demo_mode)
            if not tactile_sync_success:
                print(f"Global video synchronization failed, continuing...")
            else:
                print(f"[SUCCESS] Global video synchronization completed!")
            
            if visual_timestamp_files:
                timestamp_file = visual_timestamp_files[0]
                print(f"Using visual timestamp file after tactile sync: {timestamp_file.name}")
            else:
                print(f"[ERROR] No visual timestamp files found after tactile sync")
                return False
        
        elif visual_timestamp_files:
            if len(visual_timestamp_files) > 1:
                left_visual_file = next((f for f in visual_timestamp_files if 'left_hand_visual' in f.name), None)
                if left_visual_file:
                    timestamp_file = left_visual_file
                    print(f"Using left hand visual as time reference: {timestamp_file.name}")
                    
                    right_visual_file = next((f for f in visual_timestamp_files if 'right_hand_visual' in f.name), None)
                    if right_visual_file:
                        try:
                            left_data = load_video_timestamps(left_visual_file)
                            right_data = load_video_timestamps(right_visual_file)
                            
                            if len(left_data) == len(right_data):
                                time_diff = abs(left_data[0]['timestamp'] - right_data[0]['timestamp'])
                                if time_diff < 0.1:
                                    print(f"[SUCCESS] Dual video timestamps verified (diff: {time_diff:.3f}s)")
                                else:
                                    print(f"Large timestamp difference detected: {time_diff:.3f}s")
                            else:
                                print(f"Frame count mismatch: L={len(left_data)}, R={len(right_data)}")
                        except Exception as e:
                            print(f"Could not verify timestamp consistency: {e}")
                else:
                    timestamp_file = visual_timestamp_files[0]
                    print(f"Using visual timestamp file: {timestamp_file.name}")
            else:
                timestamp_file = visual_timestamp_files[0]
                print(f"Using visual timestamp file: {timestamp_file.name}")
        else:
            timestamp_file = timestamp_files[0]
            print(f"No visual timestamp file found, using: {timestamp_file.name}")
        video_data = load_video_timestamps(timestamp_file)
        
        video_start = video_data[0]['timestamp']
        video_end = video_data[-1]['timestamp']
        frame_count = len(video_data)
        
        print(f"Video: {frame_count} frames, {video_end - video_start:.3f}s")
        
        # Crop trajectory data
        cropped_trajectory = crop_trajectory_by_time(
            trajectory_data, video_start, video_end
        )
        
        if not cropped_trajectory:
            print(f"[ERROR] No trajectory data found for this time range")
            return False
        
        # Create pose_data directory
        pose_data_dir = demo_dir / "pose_data"
        pose_data_dir.mkdir(exist_ok=True)
        print(f"Created: {pose_data_dir}")
        
        # Process both hands
        processed_hands = []
        
        for hand in ['left', 'right']:
            print(f"\nProcessing {hand} hand...")
            
            # Generate synchronized trajectory
            hand_trajectory = generate_hand_trajectory(
                cropped_trajectory, hand, video_start, video_end, 
                frame_count, world_frame
            )
            
            # Save trajectory file
            trajectory_file = pose_data_dir / f"{hand}_hand_trajectory_synced.csv"
            save_hand_trajectory(hand_trajectory, trajectory_file)
            
            processed_hands.append(hand)
        
        # Create metadata
        metadata = {
            'demo_directory': demo_dir.name,
            'video_info': {
                'start_time': video_start,
                'end_time': video_end,
                'duration': video_end - video_start,
                'frame_count': frame_count
            },
            'trajectory_info': {
                'original_points': len(trajectory_data),
                'cropped_points': len(cropped_trajectory),
                'synchronized_points': frame_count
            },
            'processed_hands': processed_hands,
            'hands_swapped': True,
            'video_sync_info': {
                'demo_mode': demo_mode,
                'visual_sync_applied': len(visual_timestamp_files) > 1,
                'tactile_sync_applied': len(tactile_timestamp_files) > 0,
                'sync_report_available': (videos_dir / "dual_video_sync_report.json").exists(),
                'description': f'Video synchronization for {demo_mode} mode'
            },
            'created_at': datetime.now().isoformat()
        }
        
        metadata_file = pose_data_dir / "sync_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Metadata saved: {metadata_file}")
        print(f"Demo completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error processing {demo_dir.name}: {e}")
        return False

def load_session_statistics(task_dir: Path) -> int:
    """Load total demos count from session statistics file."""
    session_stats_files = list(task_dir.glob("session_statistics.json"))
    if not session_stats_files:
        print(f"[ERROR] No session_statistics.json found in {task_dir}")
        sys.exit(1)
    
    session_stats_file = session_stats_files[0]
    print(f"Loading session statistics from: {session_stats_file.name}")
    
    with open(session_stats_file, 'r') as f:
        stats = json.load(f)
    
    total_demos = stats["session_summary"]["total_demos"]
    print(f"[SUCCESS] Found {total_demos} total demos in session")
    return total_demos



def main():
    parser = argparse.ArgumentParser(
        description="Production trajectory video synchronization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 03-trajectory_video_sync.py --cfg config/data_collection.yaml
        """)  
    parser.add_argument('--cfg', type=str, required=True,
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.cfg):
        print(f"[ERROR] Configuration file not found: {args.cfg}")
        sys.exit(1)
    
    # Load configuration
    cfg = OmegaConf.load(args.cfg)
    task_name = cfg["task"]["name"]
    task_type = cfg["task"]["type"]
    
    # Determine task directory
    task_dir = Path("data") / task_name
    
    if not task_dir.exists():
        print(f"[ERROR] Task directory not found: {task_dir}")
        sys.exit(1)
    
    log_file = setup_logging(task_dir)
    print(f"Detailed log: {log_file.name}")
    
    logger.info(f"Configuration file: {args.cfg}")
    logger.info(f"Task name: {task_name}")
    logger.info(f"Task type: {task_type}")
    logger.info(f"Task directory: {task_dir}")
    
    # Load total demos count from session statistics
    print("Loading session statistics...")
    total_demos = load_session_statistics(task_dir)
    
    # Load full trajectory data
    trajectory_dir = task_dir / "all_trajectory"
    if not trajectory_dir.exists():
        print(f"[ERROR] Trajectory directory not found: {trajectory_dir}")
        sys.exit(1)
    
    trajectory_data = load_trajectory_json_files(trajectory_dir)
    traj_start = trajectory_data[0]['timestamp']
    traj_end = trajectory_data[-1]['timestamp']
    
    print(f"Trajectory: {traj_start:.3f} - {traj_end:.3f} ({len(trajectory_data)} points)")
    
    # Get default world frame
    world_frame = get_default_world_frame()
    logger.info(f"World frame: {world_frame}")

    configure_parallel_processing()

    # Find and process all demo directories
    demo_dir_ = Path(os.path.join(task_dir, "demos"))
    demo_dirs = [d for d in demo_dir_.iterdir() 
                 if d.is_dir() and d.name.startswith('demo_')]
    
    if not demo_dirs:
        print(f"[ERROR] No demo directories found")
        sys.exit(1)
    
    demo_dirs.sort()
    print(f"Found {len(demo_dirs)} demo directories")
    logger.info(f"Found demo directories: {[d.name for d in demo_dirs]}")
    
    # Validate demo count consistency
    if len(demo_dirs) != total_demos:
        warning_msg = f"Found {len(demo_dirs)} demo directories but session statistics shows {total_demos} demos"
        log_and_print(f"Warning: {warning_msg}", "warning", True)
        print(f"    Using actual demo count: {len(demo_dirs)}")
        total_demos = len(demo_dirs)
    
    # Create timestamp for files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process demos with progress bar
    print(f"\nProcessing {len(demo_dirs)} demos...")
    success_count = 0
    failed_demos = []
    
    for demo_dir in tqdm(demo_dirs, desc="Processing demos", unit="demo"):
        logger.info(f"Starting processing demo: {demo_dir.name}")
        
        success = process_demo(demo_dir, trajectory_data, world_frame, cfg)
        if success:
            success_count += 1
            logger.info(f"Demo processed successfully: {demo_dir.name}")
        else:
            failed_demos.append(demo_dir.name)
            logger.error(f"Demo processing failed: {demo_dir.name}")
    
    # Write summary
    summary = {
        'task_directory': str(task_dir),
        'processing_time': datetime.now().isoformat(),
        'log_file': str(log_file),
        'trajectory_info': {
            'source': 'Quest JSON files',
            'total_points': len(trajectory_data),
            'time_range': [traj_start, traj_end]
        },
        'results': {
            'total_demos': len(demo_dirs),
            'successful': success_count,
            'failed': len(failed_demos),
            'failed_demos': failed_demos
        }
    }
    
    summary_file = task_dir / f"processing_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total demos: {len(demo_dirs)}")
    print(f"[SUCCESS] Successful: {success_count}")
    print(f"[ERROR] Failed: {len(failed_demos)}")
    print(f"Summary saved: {summary_file.name}")
    print(f"Detailed log: {log_file.name}")
    
    if failed_demos:
        print(f"\n[ERROR] Failed demos:")
        for demo in failed_demos:
            print(f"  - {demo}")
    
    print(f"\nProcessing completed!")
    logger.info("Processing completed successfully!")
    logger.info(f"Final results: {success_count}/{len(demo_dirs)} successful")

def detect_demo_mode_from_config(config: OmegaConf) -> str:
    """
    Detect demo mode based on configuration parameters.
    
    Args:
        config: OmegaConf configuration object
        
    Returns:
        str: Demo mode description
    """
    try:
        task_type = getattr(config.task, 'type', 'single')
        enable_tactile = getattr(config.recorder, 'enable_tactile', False)
        
        print(f"Config parameters:")
        print(f"   task.type: {task_type}")
        print(f"   recorder.enable_tactile: {enable_tactile}")
        
        if task_type.lower() == 'single' and not enable_tactile:
            return "single_hand_visual_only"
        elif task_type.lower() == 'single' and enable_tactile:
            return "single_hand_with_tactile"
        elif task_type.lower() == 'bimanual' and not enable_tactile:
            return "bimanual_visual_only"
        elif task_type.lower() == 'bimanual' and enable_tactile:
            return "bimanual_with_tactile"
        else:
            print(f"Unknown config combination: task_type={task_type}, enable_tactile={enable_tactile}")
            return f"unknown_config_{task_type}_{enable_tactile}"
            
    except Exception as e:
        print(f"[ERROR] Error reading config: {e}")
        print(f"Falling back to file-based detection")
        return "config_error"

def verify_config_file_consistency(demo_mode: str, visual_files: List[Path], tactile_files: List[Path]) -> None:
    """
    Verify consistency between config and actual files
    """
    visual_count = len(visual_files)
    tactile_count = len(tactile_files)
    
    print(f"Verifying config-file consistency:")
    print(f"   Config mode: {demo_mode}")
    print(f"   Actual files: {visual_count} visual + {tactile_count} tactile")
    
    expected_combinations = {
        "single_hand_visual_only": (1, 0),
        "single_hand_with_tactile": (1, 2),
        "bimanual_visual_only": (2, 0),
        "bimanual_with_tactile": (2, 4)
    }
    
    if demo_mode in expected_combinations:
        expected_visual, expected_tactile = expected_combinations[demo_mode]
        
        if visual_count == expected_visual and tactile_count == expected_tactile:
            print(f"[SUCCESS] Config-file consistency verified!")
        else:
            print(f"Config-file mismatch:")
            print(f"     Expected: {expected_visual} visual + {expected_tactile} tactile")
            print(f"     Found: {visual_count} visual + {tactile_count} tactile")
            print(f"     Proceeding with config-based mode: {demo_mode}")
    else:
        print(f"Unknown demo mode: {demo_mode}")
        print(f"     Files found: {visual_count} visual + {tactile_count} tactile")

def detect_demo_mode(visual_files: List[Path], tactile_files: List[Path]) -> str:
    """
    Legacy function - kept for backward compatibility.
    Now uses file-based detection as fallback.
    """
    print("Using legacy file-based detect_demo_mode")
    visual_count = len(visual_files)
    tactile_count = len(tactile_files)
    
    if visual_count == 1 and tactile_count == 0:
        return "single_hand_visual_only"
    elif visual_count == 1 and tactile_count == 2:
        return "single_hand_with_tactile"
    elif visual_count == 2 and tactile_count == 0:
        return "bimanual_visual_only"
    elif visual_count == 2 and tactile_count == 4:
        return "bimanual_with_tactile"
    else:
        return f"custom_mode_v{visual_count}_t{tactile_count}"

def process_tactile_videos(demo_dir: Path, demo_mode: str) -> bool:
    """
    Process tactile videos synchronization.
    
    Args:
        demo_dir: Demo directory path
        demo_mode: Detected demo mode
        
    Returns:
        bool: Success status
    """
    try:
        videos_dir = demo_dir / "videos"
        print(f"\nPROCESSING TACTILE VIDEOS")
        print(f"{'='*50}")
        print(f"Mode: {demo_mode}")
        
        visual_files = list(videos_dir.glob("*_visual_timestamps.csv"))
        tactile_files = list(videos_dir.glob("*_tactile_timestamps.csv"))
        all_timestamp_files = visual_files + tactile_files
        
        if not tactile_files:
            print(f"[ERROR] No tactile timestamp files found")
            return False
        
        if not visual_files:
            print(f"[ERROR] No visual timestamp files found")
            return False
        
        print(f"Found {len(visual_files)} visual + {len(tactile_files)} tactile timestamp files:")
        for file in all_timestamp_files:
            print(f"   - {file.name}")
        
        print(f"\nCALCULATING GLOBAL TIME INTERSECTION")
        print(f"{'='*50}")
        
        all_time_ranges = []
        for ts_file in all_timestamp_files:
            data = load_video_timestamps(ts_file)
            start_time = data[0]['timestamp']
            end_time = data[-1]['timestamp']
            frame_count = len(data)
            
            all_time_ranges.append({
                'file': ts_file.name,
                'start': start_time,
                'end': end_time,
                'frame_count': frame_count,
                'duration': end_time - start_time
            })
            
            print(f"{ts_file.name}: {frame_count} frames, {start_time:.6f} - {end_time:.6f} ({end_time - start_time:.3f}s)")
        
        global_start = max(range_info['start'] for range_info in all_time_ranges)
        global_end = min(range_info['end'] for range_info in all_time_ranges)
        global_duration = global_end - global_start
        
        if global_start >= global_end:
            print(f"[ERROR] No time intersection found among all videos")
            return False
        
        print(f"\nGlobal time intersection:")
        print(f"   Start: {global_start:.6f}")
        print(f"   End: {global_end:.6f}")
        print(f"   Duration: {global_duration:.3f}s")
        
        visual_ref_file = visual_files[0]
        ref_data = load_video_timestamps(visual_ref_file)
        
        print(f"Using visual reference for frame alignment: {visual_ref_file.name}")
        
        print(f"\nCROPPING ALL VIDEOS TO GLOBAL INTERSECTION")
        print(f"{'='*50}")
        
        total_success_count = 0
        total_video_count = len(all_timestamp_files)
        
        video_tasks = []
        for ts_file in all_timestamp_files:
            video_name = ts_file.name.replace('_timestamps.csv', '.mp4')
            video_file = videos_dir / video_name
            
            if not video_file.exists():
                print(f"Video not found: {video_name}")
                continue
            
            video_data = load_video_timestamps(ts_file)
            start_idx = find_closest_frame_idx(video_data, global_start)
            end_idx = find_closest_frame_idx(video_data, global_end)
            
            video_tasks.append({
                'ts_file': ts_file,
                'video_file': video_file,
                'video_name': video_name,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'video_type': "Visual" if 'visual' in ts_file.name else "Tactile"
            })
        
        if len(video_tasks) <= PROCESSING_CONFIG['max_parallel_jobs']:
            total_success_count = process_videos_parallel(video_tasks, videos_dir)
        else:
            total_success_count = process_videos_batch_parallel(video_tasks, videos_dir)
        
        print(f"\nGlobal video synchronization completed: {total_success_count}/{total_video_count} successful")
        print(f"All videos now synchronized to: {global_start:.6f} - {global_end:.6f} ({global_duration:.3f}s)")
        return total_success_count > 0
        
    except Exception as e:
        print(f"[ERROR] Error processing tactile videos: {e}")
        return False

def find_closest_frame_idx(timestamps: List[Dict], target_time: float) -> int:
    """Find the closest frame index for a target timestamp."""
    min_diff = float('inf')
    closest_idx = 0
    for i, frame in enumerate(timestamps):
        diff = abs(frame['timestamp'] - target_time)
        if diff < min_diff:
            min_diff = diff
            closest_idx = i
    return closest_idx

def process_single_video_task(task: Dict, videos_dir: Path) -> bool:
    """
    Process single video task for parallel processing
    
    Args:
        task: Video processing task dictionary
        videos_dir: Video directory
        
    Returns:
        bool: Processing success status
    """
    try:
        ts_file = task['ts_file']
        video_file = task['video_file']
        video_name = task['video_name']
        start_idx = task['start_idx']
        end_idx = task['end_idx']
        video_type = task['video_type']
        
        crop_frames = end_idx - start_idx + 1
        print(f"\nProcessing {video_type}: {video_name}")
        print(f"Crop: frames {start_idx} - {end_idx} ({crop_frames} frames)")
        
        temp_video = videos_dir / f"{video_name.replace('.mp4', '_temp.mp4')}"
        if crop_video_ffmpeg_simple(video_file, temp_video, start_idx, end_idx):
            temp_video.replace(video_file)
            print(f"[SUCCESS] Video cropped: {video_name}")
            
            if update_timestamps_csv_simple(ts_file, start_idx, end_idx):
                print(f"[SUCCESS] Timestamps updated: {ts_file.name}")
                return True
            else:
                print(f"[ERROR] Failed to update timestamps: {ts_file.name}")
                return False
        else:
            print(f"[ERROR] Failed to crop video: {video_name}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error processing video task: {e}")
        return False

def process_videos_parallel(video_tasks: List[Dict], videos_dir: Path) -> int:
    """
    Process multiple videos in parallel
    
    Args:
        video_tasks: List of video processing tasks
        videos_dir: Video directory
        
    Returns:
        int: Number of successfully processed videos
    """
    print(f"Processing {len(video_tasks)} videos in parallel (max {PROCESSING_CONFIG['max_parallel_jobs']} jobs)")
    
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=PROCESSING_CONFIG['max_parallel_jobs']) as executor:
        future_to_task = {
            executor.submit(process_single_video_task, task, videos_dir): task 
            for task in video_tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                    print(f"[SUCCESS] Completed: {task['video_name']}")
                else:
                    print(f"[ERROR] Failed: {task['video_name']}")
            except Exception as e:
                print(f"[ERROR] Exception processing {task['video_name']}: {e}")
    
    return success_count

def process_videos_batch_parallel(video_tasks: List[Dict], videos_dir: Path) -> int:
    """
    Process large number of videos in parallel batches
    
    Args:
        video_tasks: List of video processing tasks
        videos_dir: Video directory
        
    Returns:
        int: Number of successfully processed videos
    """
    batch_size = PROCESSING_CONFIG['max_parallel_jobs']
    total_success = 0
    total_batches = (len(video_tasks) + batch_size - 1) // batch_size
    
    print(f"Processing {len(video_tasks)} videos in {total_batches} batches (batch size: {batch_size})")
    
    for i in range(0, len(video_tasks), batch_size):
        batch = video_tasks[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} videos)")
        batch_success = process_videos_parallel(batch, videos_dir)
        total_success += batch_success
        
        print(f"Batch {batch_num} completed: {batch_success}/{len(batch)} successful")
    
    return total_success

def get_video_fps(video_path: Path) -> float:
    """
    Get video frame rate
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-select_streams', 'v:0', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            info = json.loads(result.stdout)
            streams = info.get('streams', [])
            if streams:
                fps_str = streams[0].get('r_frame_rate', '30/1')
                num, den = fps_str.split('/')
                return float(num) / float(den)
        
        print(f"Could not detect FPS, using default 30fps")
        return 30.0
        
    except Exception as e:
        print(f"FPS detection failed: {e}, using default 30fps")
        return 30.0

def crop_video_ffmpeg_simple(input_video: Path, output_video: Path, start_frame: int, end_frame: int) -> bool:
    """
    Crop video using FFmpeg with CPU encoding only.
    """
    try:
        fps = get_video_fps(input_video)
        
        start_time = start_frame / fps
        frame_count = end_frame - start_frame + 1
        duration = frame_count / fps
        
        print(f"Cropping {input_video.name}:")
        print(f"   Frames: {start_frame}-{end_frame} ({frame_count} frames)")
        print(f"   Time: {start_time:.3f}s - {start_time + duration:.3f}s ({duration:.3f}s)")
        print(f"   FPS: {fps:.2f}")
        print(f"   Using: CPU Encoding")
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_video),
            '-ss', f'{start_time:.6f}',
            '-t', f'{duration:.6f}',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'copy',
            '-avoid_negative_ts', 'make_zero',
            str(output_video)
        ]
        
        print(f"Running: ffmpeg -y -i {input_video.name} -ss {start_time:.3f} -t {duration:.3f} -c:v libx264...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[ERROR] FFmpeg failed: {result.stderr}")
            logger.error(f"FFmpeg failed for {input_video.name}: {result.stderr}")
            return False
        
        if not output_video.exists():
            print(f"[ERROR] Output video not created: {output_video}")
            return False
        
        print(f"[SUCCESS] Video cropped successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error cropping video: {e}")
        logger.error(f"Video cropping error for {input_video.name}: {e}")
        return False



def update_timestamps_csv_simple(timestamp_file: Path, start_idx: int, end_idx: int) -> bool:
    """
    Update timestamp CSV file (simplified version for main script).
    """
    try:
        with open(timestamp_file, 'r') as f:
            reader = csv.DictReader(f)
            original_data = list(reader)
        
        cropped_data = original_data[start_idx:end_idx + 1]
        
        for i, row in enumerate(cropped_data):
            row['frame_number'] = str(i + 1)
        
        with open(timestamp_file, 'w', newline='') as f:
            if cropped_data:
                writer = csv.DictWriter(f, fieldnames=cropped_data[0].keys())
                writer.writeheader()
                writer.writerows(cropped_data)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error updating timestamps: {e}")
        return False

if __name__ == "__main__":
    main() 