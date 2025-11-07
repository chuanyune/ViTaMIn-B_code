#!/usr/bin/env python3
"""
Convert bimanual dataset to two separate single-hand datasets.

This script takes a bimanual dataset and creates two separate single-hand datasets:
- One for left hand only 
- One for right hand only

The script maintains correct hand-to-sensor correspondence:
- Left hand dataset: left_hand_visual.mp4 + left_hand_*_tactile.mp4 + left hand trajectory/gripper data
- Right hand dataset: right_hand_visual.mp4 + right_hand_*_tactile.mp4 + right hand trajectory/gripper data

This ensures the detection logic in 07_generate_dataset_plan.py correctly identifies
each as single-hand mode with proper sensor mapping.
"""

import os
import shutil
import argparse
from pathlib import Path
import pandas as pd
import json

def copy_demo_structure(src_demo_dir, dst_demo_dir, hand='left'):
    """
    Copy demo directory structure for single hand mode.
    
    Args:
        src_demo_dir: Source demo directory (bimanual)
        dst_demo_dir: Destination demo directory (single hand)
        hand: 'left' or 'right' - which hand to keep
    """
    dst_demo_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy pose_data directory with only the target hand
    src_pose_dir = src_demo_dir / 'pose_data'
    dst_pose_dir = dst_demo_dir / 'pose_data'
    dst_pose_dir.mkdir(exist_ok=True)
    
    # Copy trajectory file for the target hand
    src_traj = src_pose_dir / f'{hand}_hand_trajectory_synced.csv'
    if src_traj.exists():
        shutil.copy2(src_traj, dst_pose_dir)
        print(f"    Copied trajectory: {src_traj.name}")
    else:
        print(f"    WARNING: Missing trajectory file: {src_traj}")
        return False
    
    # Copy metadata files if they exist
    for meta_file in ['left_metadata.json', 'right_metadata.json']:
        src_meta = src_pose_dir / meta_file
        if src_meta.exists():
            shutil.copy2(src_meta, dst_pose_dir)
    
    # 2. Copy gripper width file for the target hand
    src_gripper = src_demo_dir / f'gripper_width_{hand}.csv'
    dst_gripper = dst_demo_dir / f'gripper_width_{hand}.csv'
    if src_gripper.exists():
        shutil.copy2(src_gripper, dst_gripper)
        print(f"    Copied gripper: {src_gripper.name}")
    else:
        print(f"    WARNING: Missing gripper file: {src_gripper}")
        return False
    
    # 3. Copy videos directory with correct naming for single hand mode
    src_videos_dir = src_demo_dir / 'videos'
    dst_videos_dir = dst_demo_dir / 'videos'
    dst_videos_dir.mkdir(exist_ok=True)
    
    # Copy visual video with correct naming based on which hand we're keeping
    if hand == 'left':
        src_visual = src_videos_dir / 'left_hand_visual.mp4'
        dst_visual = dst_videos_dir / 'left_hand_visual.mp4'  # Left hand keeps left naming
    else:
        src_visual = src_videos_dir / 'right_hand_visual.mp4'
        dst_visual = dst_videos_dir / 'right_hand_visual.mp4'  # Right hand keeps right naming
    
    if src_visual.exists():
        shutil.copy2(src_visual, dst_visual)
        print(f"    Copied visual: {src_visual.name} -> {dst_visual.name}")
    else:
        print(f"    WARNING: Missing visual video: {src_visual}")
        return False
    
    # Copy tactile videos with correct naming - no mapping, keep original hand correspondence
    tactile_files = [
        (f'{hand}_hand_left_tactile.mp4', f'{hand}_hand_left_tactile.mp4'),
        (f'{hand}_hand_right_tactile.mp4', f'{hand}_hand_right_tactile.mp4')
    ]
    
    for src_name, dst_name in tactile_files:
        src_tactile = src_videos_dir / src_name
        dst_tactile = dst_videos_dir / dst_name
        if src_tactile.exists():
            shutil.copy2(src_tactile, dst_tactile)
            print(f"    Copied tactile: {src_name} -> {dst_name}")
    
    # 4. Copy ArUco detection files with correct naming
    if hand == 'left':
        src_aruco = src_demo_dir / 'tag_detection_left.pkl'
        dst_aruco = dst_demo_dir / 'tag_detection_left.pkl'  # Left hand keeps left naming
    else:
        src_aruco = src_demo_dir / 'tag_detection_right.pkl'
        dst_aruco = dst_demo_dir / 'tag_detection_right.pkl'  # Right hand keeps right naming
    
    if src_aruco.exists():
        shutil.copy2(src_aruco, dst_aruco)
        print(f"    Copied ArUco: {src_aruco.name} -> {dst_aruco.name}")
    else:
        print(f"    WARNING: Missing ArUco file: {src_aruco}")
        return False
    
    # 5. Copy tactile_points directory if it exists
    src_tactile_points = src_demo_dir / 'tactile_points'
    if src_tactile_points.exists():
        dst_tactile_points = dst_demo_dir / 'tactile_points'
        
        # Copy tactile point files with correct naming - no mapping, keep original hand correspondence
        for src_file in src_tactile_points.glob('*'):
            if src_file.is_file():
                if f'{hand}_hand' in src_file.name:
                    # Keep files that belong to the target hand with original naming
                    dst_name = src_file.name  # No renaming needed - keep original hand correspondence
                    
                    dst_tactile_points.mkdir(exist_ok=True)
                    dst_file = dst_tactile_points / dst_name
                    shutil.copy2(src_file, dst_file)
                    print(f"    Copied tactile points: {src_file.name} -> {dst_name}")
    
    return True

def convert_bimanual_to_single(src_task_dir, dst_base_dir):
    """
    Convert a bimanual task dataset to two single-hand datasets.
    
    Args:
        src_task_dir: Path to source bimanual task directory
        dst_base_dir: Base directory where to create the two single-hand datasets
    """
    src_path = Path(src_task_dir)
    dst_base = Path(dst_base_dir)
    
    if not src_path.exists():
        print(f"ERROR: Source directory does not exist: {src_path}")
        return
    
    task_name = src_path.name
    
    # Create output directories
    left_task_dir = dst_base / f"{task_name}_left_only"
    right_task_dir = dst_base / f"{task_name}_right_only"
    
    # Create demos directories
    left_demos_dir = left_task_dir / 'demos'
    right_demos_dir = right_task_dir / 'demos'
    
    src_demos_dir = src_path / 'demos'
    
    if not src_demos_dir.exists():
        print(f"ERROR: Source demos directory does not exist: {src_demos_dir}")
        return
    
    # Find all demo directories
    demo_dirs = sorted([d for d in src_demos_dir.iterdir() if d.is_dir() and d.name.startswith('demo_')])
    
    if not demo_dirs:
        print(f"ERROR: No demo directories found in {src_demos_dir}")
        return
    
    print(f"Found {len(demo_dirs)} demo directories")
    print(f"Creating left-hand dataset: {left_task_dir}")
    print(f"Creating right-hand dataset: {right_task_dir}")
    
    left_success = 0
    right_success = 0
    
    # Process each demo
    for demo_dir in demo_dirs:
        print(f"\nProcessing {demo_dir.name}:")
        
        # Create left-hand version
        left_dst = left_demos_dir / demo_dir.name
        print(f"  Creating left-hand version...")
        if copy_demo_structure(demo_dir, left_dst, 'left'):
            left_success += 1
            print(f"  ✅ Left-hand version created successfully")
        else:
            print(f"  ❌ Failed to create left-hand version")
        
        # Create right-hand version  
        right_dst = right_demos_dir / demo_dir.name
        print(f"  Creating right-hand version...")
        if copy_demo_structure(demo_dir, right_dst, 'right'):
            right_success += 1
            print(f"  ✅ Right-hand version created successfully")
        else:
            print(f"  ❌ Failed to create right-hand version")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total demos processed: {len(demo_dirs)}")
    print(f"Left-hand demos created: {left_success}")
    print(f"Right-hand demos created: {right_success}")
    print(f"\nOutput directories:")
    print(f"  Left-hand dataset: {left_task_dir}")
    print(f"  Right-hand dataset: {right_task_dir}")
    print(f"\nNext steps:")
    print(f"1. Run 07_generate_dataset_plan.py on each new dataset")
    print(f"2. Run 08_generate_replay_buffer.py on each new dataset")

def main():
    parser = argparse.ArgumentParser(
        description="Convert bimanual dataset to two single-hand datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_bimanual_to_single.py --src /home/drj/codehub/ViTaMIn-B/data/_927_vb_sensor_pick_and_place --dst data/
  
This will create:
  data/_926_alltact_pick_and_place_left_only/
  data/_926_alltact_pick_and_place_right_only/
        """)
    
    parser.add_argument('--src', type=str, required=True, 
                       help='Source bimanual task directory (e.g., data/_926_alltact_pick_and_place)')
    parser.add_argument('--dst', type=str, required=True,
                       help='Destination base directory (e.g., data/)')
    
    args = parser.parse_args()
    
    convert_bimanual_to_single(args.src, args.dst)

if __name__ == "__main__":
    main()
