import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import multiprocessing
import subprocess
import concurrent.futures
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf

def main(config_file):
    cfg = OmegaConf.load(config_file)
    task_name = cfg["task"]["name"]
    task_type = cfg["task"]["type"]
    num_workers = cfg["calculate_width"]["max_workers"]
    camera_intrinsics = cfg["calculate_width"]["cam_intrinsic_json_path"]
    aruco_yaml = cfg["calculate_width"]["aruco_yaml"]
    folder = os.path.join("data", task_name, "demos")   
    input_video_dirs = []
    
    # Find demo directories that contain the video files we need
    for root, dirs, files in os.walk(folder):
        # Only process demo directories (not subdirectories)
        if os.path.basename(root).startswith('demo_'):
            demo_path = pathlib.Path(root)
            videos_dir = demo_path.joinpath('videos')
            
            # Check if this demo directory contains the videos we need for ArUco detection
            has_target_video = False
            
            if task_type == "single":
                # For single mode, only look for left hand video
                if videos_dir.joinpath('left_hand_visual.mp4').exists():
                    has_target_video = True
            elif task_type == "bimanual":
                # For bimanual, look for either left or right hand video
                if (videos_dir.joinpath('left_hand_visual.mp4').exists() or 
                    videos_dir.joinpath('right_hand_visual.mp4').exists()):
                    has_target_video = True
            
            if has_target_video:
                input_video_dirs.append(root)
    
    # Remove duplicates and sort
    input_video_dirs = sorted(list(set(input_video_dirs)))
    print(f'Found {len(input_video_dirs)} video dirs with target videos')
    assert os.path.isfile(camera_intrinsics)

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    script_path = pathlib.Path(__file__).parent.parent.joinpath('utils', 'detect_aruco.py')

    with tqdm(total=len(input_video_dirs)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            future_to_info = {}  # Record processing info for each future
            for video_dir in tqdm(input_video_dirs):
                demo_dir = pathlib.Path(video_dir)
                videos_dir = demo_dir.joinpath('videos')
                
                # Fix dual-hand mode: process all existing video files
                videos_to_process = []
                
                if task_type == "single":
                    # Single-hand mode: only process left hand video
                    if videos_dir.joinpath('left_hand_visual.mp4').exists():
                        videos_to_process.append({
                            'video_path': videos_dir.joinpath('left_hand_visual.mp4'),
                            'pkl_filename': 'tag_detection_left.pkl'
                        })
                elif task_type == "bimanual":
                    # Dual-hand mode: process all existing video files
                    if videos_dir.joinpath('left_hand_visual.mp4').exists():
                        videos_to_process.append({
                            'video_path': videos_dir.joinpath('left_hand_visual.mp4'),
                            'pkl_filename': 'tag_detection_left.pkl'
                        })
                    if videos_dir.joinpath('right_hand_visual.mp4').exists():
                        videos_to_process.append({
                            'video_path': videos_dir.joinpath('right_hand_visual.mp4'),
                            'pkl_filename': 'tag_detection_right.pkl'
                        })
                else:
                    print(f"Unknown task type: {task_type}")
                    continue
                
                if not videos_to_process:
                    print(f"Warning: No valid video file found in {videos_dir}")
                    continue
                
                # Create independent processing task for each video file
                for video_info in videos_to_process:
                    video_path = video_info['video_path']
                    pkl_filename = video_info['pkl_filename']
                    
                    # Output pkl file to demo root directory, not videos subdirectory
                    pkl_path = demo_dir.joinpath(pkl_filename)
                    # if pkl_path.is_file():
                    #     print(f"tag_detection.pkl already exists, skipping {video_dir.name}")
                    #     continue

                    # run SLAM
                    cmd = [
                        sys.executable, script_path,
                        '--input', str(video_path),
                        '--output', str(pkl_path),
                        '--intrinsics_json', camera_intrinsics,
                        '--aruco_yaml', str(aruco_yaml),
                        '--num_workers', str(num_workers)
                    ]

                    if len(futures) >= num_workers:
                        # limit number of inflight tasks
                        completed, futures = concurrent.futures.wait(futures, 
                            return_when=concurrent.futures.FIRST_COMPLETED)
                        pbar.update(len(completed))

                    future = executor.submit(
                        lambda x: subprocess.run(x, 
                            capture_output=True), 
                        cmd)
                    futures.add(future)
                    # Record processing info for corresponding future
                    future_to_info[future] = {
                        'video_dir': video_dir,
                        'demo_dir': demo_dir,
                        'pkl_path': pkl_path,
                        'pkl_filename': pkl_filename,
                        'video_path': str(video_path)  # Add video path info
                    }
                    # futures.add(executor.submit(lambda x: print(' '.join(x)), cmd))

            completed, futures = concurrent.futures.wait(futures)            
            pbar.update(len(completed))

    # Statistics of processing results
    print("=" * 80)
    print("ArUco detection processing completed! Statistics:")
    print("=" * 80)
    
    successful_count = 0
    failed_count = 0
    successful_demos = []
    failed_demos = []
    
    # Analyze execution results of each task
    for future in completed:
        result = future.result()
        info = future_to_info.get(future, {})
        demo_dir = info.get('demo_dir')
        pkl_path = info.get('pkl_path')
        
        # Check subprocess return code to determine success
        if result.returncode == 0:
            successful_count += 1
            if demo_dir:
                successful_demos.append(str(demo_dir))
        else:
            failed_count += 1
            if demo_dir:
                failed_demos.append(str(demo_dir))
            # Print error information
            if result.stderr:
                error_msg = result.stderr.decode().strip()
                if error_msg:
                    video_path = info.get('video_path', 'unknown')
                    print(f"[ERROR] Error output ({demo_dir} - {video_path}): {error_msg}")
    
    total_tasks = len(completed)  # Total tasks (may be more than demo count, as dual-hand mode generates 2 tasks per demo)
    print(f"Total processing: {len(input_video_dirs)} demo directories, {total_tasks} video tasks")
    print(f"[SUCCESS] Successfully processed: {successful_count}")
    print(f"[ERROR] Processing failed: {failed_count}")
    print(f"Success rate: {successful_count/total_tasks*100:.1f}%" if total_tasks > 0 else "Success rate: 0%")
    
    # Count actually generated files
    actual_pkl_files = []
    for video_dir in input_video_dirs:
        demo_dir = pathlib.Path(video_dir)
        
        # Check corresponding pkl files based on task type
        if task_type == "single":
            pkl_path = demo_dir.joinpath('tag_detection_left.pkl')
            if pkl_path.exists():
                actual_pkl_files.append(str(pkl_path))
        elif task_type == "bimanual":
            left_pkl = demo_dir.joinpath('tag_detection_left.pkl')
            right_pkl = demo_dir.joinpath('tag_detection_right.pkl')
            if left_pkl.exists():
                actual_pkl_files.append(str(left_pkl))
            if right_pkl.exists():
                actual_pkl_files.append(str(right_pkl))
    
    if len(actual_pkl_files) > 0:
        print(f"\nActually generated {len(actual_pkl_files)} pkl files:")
        for pkl_file in sorted(actual_pkl_files):
            file_size = pathlib.Path(pkl_file).stat().st_size / 1024  # KB
            print(f"   {pkl_file} ({file_size:.1f} KB)")
    
    # Find missing files
    missing_dirs = []
    for video_dir in input_video_dirs:
        demo_dir = pathlib.Path(video_dir)
        
        # Check if expected pkl files are missing
        has_expected_pkl = False
        if task_type == "single":
            if demo_dir.joinpath('tag_detection_left.pkl').exists():
                has_expected_pkl = True
        elif task_type == "bimanual":
            if (demo_dir.joinpath('tag_detection_left.pkl').exists() or 
                demo_dir.joinpath('tag_detection_right.pkl').exists()):
                has_expected_pkl = True
        
        if not has_expected_pkl:
            missing_dirs.append(video_dir)
    
    if len(missing_dirs) > 0:
        print(f"\n{len(missing_dirs)} directories missing pkl files:")
        for video_dir in missing_dirs:
            print(f"  ✗ {video_dir}")
    
    print("=" * 80)

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Task name')
    args = parser.parse_args()
    config_file = args.cfg
    main(config_file)
