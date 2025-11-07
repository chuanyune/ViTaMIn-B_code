# %%
import sys
import os
from pathlib import Path
project_root = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(project_root))

import pickle
import argparse
import numpy as np
import cv2
from omegaconf import OmegaConf
import concurrent.futures
import multiprocessing as mp
import pathlib
from tqdm import tqdm

def farthest_point_sampling(points, num_samples):
    if len(points) == 0:
        return np.array([])
    
    points = np.array(points)
    if len(points) <= num_samples:
        return points
    
    # Initialize
    N = len(points)
    selected_indices = []
    distances = np.full(N, np.inf)
    
    # Randomly select first point
    first_idx = np.random.randint(0, N)
    selected_indices.append(first_idx)
    
    # Update distances
    distances = np.minimum(distances, np.linalg.norm(points - points[first_idx], axis=1))
    
    # Iteratively select remaining points
    for _ in range(num_samples - 1):
        # Select point farthest from selected points
        farthest_idx = np.argmax(distances)
        selected_indices.append(farthest_idx)
        
        # Update minimum distances to selected points
        new_distances = np.linalg.norm(points - points[farthest_idx], axis=1)
        distances = np.minimum(distances, new_distances)
    
    return points[selected_indices]


def get_point(video_address):
    # Optimize video capture for better performance
    cap = cv2.VideoCapture(video_address)
    
    # Set buffer size to reduce memory usage and improve performance
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    K = np.asmatrix(
        [[506.36, 0, 309.94],
         [0,  506.54, 239.48],
         [0, 0, 1]]
    )
    v0, u0 = K[0,2], K[1,2]
    fv, fu = K[0,0], K[1,1]

    point_cloud = []
    point_cloud_flow = []

    is_center_calibrated = False
    x_center_err = 0
    y_center_err = 0
    
    # Set threshold parameters based on video filename
    video_filename = os.path.basename(video_address)
    if "right_hand_right_tactile" in video_filename:
        lower_bound = 25
        higher_bound = 255
    elif "right_hand_left_tactile" in video_filename:
        lower_bound = 40
        higher_bound = 255
    elif "left_hand_right_tactile" in video_filename:
        lower_bound = 25
        higher_bound = 255
    elif "left_hand_left_tactile" in video_filename:
        lower_bound = 25
        higher_bound = 255


    while True:
        ret, frame = cap.read(cv2.CAP_PROP_FPS)

        if not ret:
            print("End of Video")
            break

        # Get edge
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (3,3), 50)
        _, bin_frame = cv2.threshold(gray_frame, lower_bound, higher_bound, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #Contour Filter
        contours_large = []
        for contour in contours:
            if len(contour) > 500:
                contours_large.append(contour)


        contours_cutted = []
        for contour in contours_large:
            contours_cutted.append([point for point in contour if 10 < point[0][1] < 470])

        contours_filtered = []
        smooth_len = 4
        for contour in contours_cutted:
            contours_filtered.append([])
            for i in range(smooth_len,len(contour)-smooth_len):

                point_avg = np.zeros((1,2))
                for j in range(0,2 * smooth_len + 1):
                    point_avg += contour[i+j-smooth_len]
                point_avg /= 2 * smooth_len + 1

                if np.linalg.norm(point_avg - contour[i]) < 5:
                    contours_filtered[-1].append(point_avg.astype(np.int32))
                else:
                    contours_filtered[-1].append(contour[i])

        #Calibrate the center

        if is_center_calibrated == False:

            # There are two coordinate systems: optical center system and frame center system. 
            # We need to find the transformation between them.

            # Calibration method:
            # Calculate the pixel coordinates of the frame center, treating it as the projection of the frame center system on the imaging plane.
            u_center = 0
            v_center = 0
            point_num = 0
            #for contour in contours_filtered:
            for contour in contours_large:
                for point_ori in contour:
                    u = point_ori[0][1]
                    v = point_ori[0][0]
                    u_center += u
                    v_center += v
                    point_num += 1
            u_center /= point_num
            v_center /= point_num

            # Calculate the pixel coordinate deviation of frame center system relative to optical center system
            u_center_err = u_center - u0
            v_center_err = v_center - v0

            # Since the xOy plane of frame center system coincides with the optical center system, 
            # we can calculate the actual deviation vector by intersecting the deviation vector with the z=30 plane (bottom of the rubber pad)
            z_beneath = 40
            x_center_err = u_center_err * z_beneath/fu
            y_center_err = v_center_err * z_beneath/fv

            is_center_calibrated = True

        else: pass


        #Calculate Point Clouds
        for contour in contours_filtered:

            # Side plane equation in frame center system calculated from CAD parameters (not optical center system)
            # Rubber pad bottom is 30mm from camera, plane reference point A coordinates (15,16.5,30)
            nx, ny, nz = 0, 33, 4
            Ax, Ay, Az = 15, 16.5, 30

            # Transform center system equation to optical center system. err vector points from optical center origin to frame center origin.
            nx1, ny1, nz1 = nx + x_center_err, ny + y_center_err, nz
            Ax1, Ay1, Az1 = Ax + x_center_err, Ay + y_center_err, Az
            b1 = Ax1 * nx1 + Ay1 * ny1 + Az1 * nz1

            nx2, ny2, nz2 = nx + x_center_err, - ny + y_center_err, nz
            Ax2, Ay2, Az2 = Ax + x_center_err, - Ay + y_center_err, Az
            b2 = Ax2 * nx2 + Ay2 * ny2 + Az2 * nz2

            for point_ori in contour:
                u = point_ori[0][1] - u0
                v = point_ori[0][0] - v0

                if v > 0:
                    z = (b1/(nx1/fu * u + ny1/fv * v + nz1)).astype('float32')
                    x = (z/fu * u).astype('float32')
                    y = (z/fv * v).astype('float32')

                if v < 0:
                    z = (b2/(nx2/fu * u + ny2/fv * v + nz2)).astype('float32')
                    x = (z/fu * u).astype('float32')
                    y = (z/fv * v).astype('float32')

                if not (abs(z) > 63):
                    point_cloud.append([x,y,z])

        point_cloud_flow.append(point_cloud.copy())
        point_cloud.clear()

    cap.release()

    return point_cloud_flow


def find_tactile_videos(demo_dir, task_type, use_tactile):
    videos_dir = demo_dir / 'videos'
    tactile_videos = {}
    
    if not use_tactile or not videos_dir.exists():
        return tactile_videos
    
    for video_file in videos_dir.glob('*tactile*.mp4'):
        if video_file.is_file():
            video_key = video_file.stem
            tactile_videos[video_key] = video_file
    
    return tactile_videos


def process_single_video_task(task_info):
    demo_dir, video_name, video_path, fps_num_points = task_info
    
    try:
        # Create output directory
        output_dir = demo_dir / 'tactile_points'
        output_dir.mkdir(exist_ok=True)
        
        # Use get_point function with video path
        all_frames_points = get_point(str(video_path))
        
        # Apply FPS sampling to each frame to ensure fixed point count
        fps_frames_points = []
        for frame_points in all_frames_points:
            if len(frame_points) > 0:
                sampled_points = farthest_point_sampling(frame_points, fps_num_points)
                # If insufficient points, pad with zeros; if too many, truncate
                if len(sampled_points) < fps_num_points:
                    # Pad with zero points
                    padding = np.zeros((fps_num_points - len(sampled_points), 3))
                    sampled_points = np.vstack([sampled_points, padding])
                elif len(sampled_points) > fps_num_points:
                    sampled_points = sampled_points[:fps_num_points]
                fps_frames_points.append(sampled_points)
            else:
                # No contact, fill with zeros
                fps_frames_points.append(np.zeros((fps_num_points, 3)))
        all_frames_points = fps_frames_points
        
        # Output filename (without fps suffix)
        output_filename = f"{video_name}_points.npy"
        output_path = output_dir / output_filename
        
        # Convert to fixed-shape numpy array and save
        points_array = np.array(all_frames_points)  # shape: (num_frames, fps_num_points, 3)
        np.save(output_path, points_array)
        
        # Calculate statistics
        total_frames = len(all_frames_points)
        # Count non-zero points (excluding padded zeros)
        total_points = 0
        non_empty_frames = 0
        for frame_points in all_frames_points:
            non_zero_points = np.count_nonzero(np.any(frame_points != 0, axis=1))
            if non_zero_points > 0:
                total_points += non_zero_points
                non_empty_frames += 1
        
        return {
            'output_path': str(output_path),
            'demo_name': demo_dir.name,
            'video_name': video_name,
            'total_frames': total_frames,
            'total_points': total_points,
            'valid_frames': non_empty_frames,
            'file_size': output_path.stat().st_size / 1024,
            'fps_num_points': fps_num_points
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'demo_name': demo_dir.name,
            'video_name': video_name
        }


def process_tactile_video(video_path, output_dir, video_name, fps_num_points=256):
    try:
        # Use new get_point function, directly pass video path
        all_frames_points = get_point(str(video_path))
        
        # Apply FPS sampling to each frame point cloud to ensure fixed point count
        fps_frames_points = []
        for frame_points in all_frames_points:
            if len(frame_points) > 0:
                sampled_points = farthest_point_sampling(frame_points, fps_num_points)
                # If insufficient points, pad with zeros; if too many, truncate
                if len(sampled_points) < fps_num_points:
                    # Pad with zero points
                    padding = np.zeros((fps_num_points - len(sampled_points), 3))
                    sampled_points = np.vstack([sampled_points, padding])
                elif len(sampled_points) > fps_num_points:
                    sampled_points = sampled_points[:fps_num_points]
                fps_frames_points.append(sampled_points)
            else:
                # No contact, fill with zeros
                fps_frames_points.append(np.zeros((fps_num_points, 3)))
        all_frames_points = fps_frames_points
        
        # Output filename (without fps suffix)
        output_filename = f"{video_name}_points.npy"
        output_path = output_dir / output_filename
        
        # Convert to fixed-shape numpy array and save
        points_array = np.array(all_frames_points)  # shape: (num_frames, fps_num_points, 3)
        np.save(output_path, points_array)
        
        # Calculate statistics
        total_frames = len(all_frames_points)
        # Count non-zero points (excluding padded zeros)
        total_points = 0
        non_empty_frames = 0
        for frame_points in all_frames_points:
            non_zero_points = np.count_nonzero(np.any(frame_points != 0, axis=1))
            if non_zero_points > 0:
                total_points += non_zero_points
                non_empty_frames += 1
        
        return {
            'output_path': str(output_path),
            'video_name': video_name,
            'total_frames': total_frames,
            'total_points': total_points,
            'valid_frames': non_empty_frames,
            'file_size': output_path.stat().st_size / 1024,
            'fps_num_points': fps_num_points
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'video_name': video_name
        }


def process_demo_directory(demo_dir, task_type, use_tactile, max_workers=4, fps_num_points=256):
    tactile_videos = find_tactile_videos(demo_dir, task_type, use_tactile)
    
    if not tactile_videos:
        return []
    
    output_dir = demo_dir / 'tactile_points'
    output_dir.mkdir(exist_ok=True)
    
    successful_outputs = []
    
    # Create demo-level progress bar
    demo_pbar = tqdm(total=len(tactile_videos), 
                     desc=f"Demo {demo_dir.name}", 
                     unit="videos",
                     position=1,
                     leave=False)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {}
        for video_name, video_path in tactile_videos.items():
            future = executor.submit(process_tactile_video, video_path, output_dir, video_name, fps_num_points)
            future_to_video[future] = (video_name, video_path)
        
        for future in concurrent.futures.as_completed(future_to_video):
            video_name, video_path = future_to_video[future]
            try:
                result = future.result()
                if 'error' not in result:
                    successful_outputs.append(result['output_path'])
                    fps_info = f", FPS({result['fps_num_points']})"
                    tqdm.write(f"  [SUCCESS] {result['video_name']}: "
                              f"{result['total_frames']} frames, "
                              f"{result['valid_frames']} valid, "
                              f"{result['file_size']:.1f} KB{fps_info}")
                else:
                    tqdm.write(f"  [ERROR] {result['video_name']}: {result['error']}")
            except Exception as e:
                tqdm.write(f"  [ERROR] {video_name}: {str(e)}")
            
            demo_pbar.update(1)
    
    demo_pbar.close()
    return successful_outputs


def main(config_file):
    cfg = OmegaConf.load(config_file)
    task_name = cfg["task"]["name"]
    task_type = cfg["task"]["type"]
    # Point cloud extraction script only cares about point cloud processing
    use_tactile_pc = cfg["output_train_data"]["use_tactile_pc"]
    # For compatibility, also check old config
    use_tactile_old = cfg["output_train_data"].get("use_tactile", False)
    use_tactile = use_tactile_pc or use_tactile_old
    
    # Optimize worker count for multiprocessing
    config_workers = cfg["tactile_point_extraction"]["max_workers"]
    cpu_count = mp.cpu_count()
    # Use configured workers or CPU count, whichever is smaller (avoid oversubscription)
    max_workers = min(config_workers, cpu_count) if config_workers > 0 else cpu_count
    
    # Read FPS point count configuration
    fps_num_points = cfg["tactile_point_extraction"]["fps_num_points"]
    
    print("=" * 80)
    print("Tactile Point Cloud Extraction Tool (Multiprocessing Optimized)")
    print("=" * 80)
    print(f"Task name: {task_name}")
    print(f"Task type: {task_type}")
    print(f"Tactile processing: {'enabled' if use_tactile else 'disabled'}")
    print(f"Available CPU cores: {cpu_count}")
    print(f"Using processes: {max_workers}")
    print(f"FPS target points: {fps_num_points}")
    
    if not use_tactile:
        print("Tactile processing is disabled, exiting")
        return
    
    demos_folder = os.path.join("data", task_name, "demos")
    if not os.path.exists(demos_folder):
        print(f"Demo directory does not exist: {demos_folder}")
        return
    
    demo_dirs = []
    for root, dirs, files in os.walk(demos_folder):
        if os.path.basename(root).startswith('demo_'):
            demo_dirs.append(pathlib.Path(root))
    
    demo_dirs = sorted(demo_dirs)
    print(f"Found {len(demo_dirs)} demo directories")
    
    if not demo_dirs:
        print("No demo directories found")
        return
    
    stats = {
        'total_demos': len(demo_dirs),
        'processed_demos': 0,
        'total_videos': 0,
        'successful_videos': 0,
        'total_output_files': 0
    }
    
    # Collect all video tasks
    all_video_tasks = []
    demo_video_count = {}
    
    print("Collecting all video tasks...")
    for demo_dir in demo_dirs:
        tactile_videos = find_tactile_videos(demo_dir, task_type, use_tactile)
        demo_video_count[demo_dir.name] = len(tactile_videos)
        stats['total_videos'] += len(tactile_videos)
        
        for video_name, video_path in tactile_videos.items():
            task_info = (demo_dir, video_name, video_path, fps_num_points)
            all_video_tasks.append(task_info)
    
    print(f"Total collected {len(all_video_tasks)} video tasks from {len(demo_dirs)} demos")
    
    if not all_video_tasks:
        print("No videos found for processing")
        return
    
    all_output_files = []
    processed_demos = set()
    
    # Process all videos in parallel across demos using multiprocessing
    print(f"Using {max_workers} CPU processes to process all videos in parallel...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {}
        for task_info in all_video_tasks:
            future = executor.submit(process_single_video_task, task_info)
            future_to_task[future] = task_info
        
        # Create overall progress bar
        main_pbar = tqdm(total=len(all_video_tasks), desc="Processing all videos", unit="video")
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_task):
            task_info = future_to_task[future]
            demo_dir, video_name, video_path, _ = task_info
            
            try:
                result = future.result()
                if 'error' not in result:
                    all_output_files.append(result['output_path'])
                    stats['successful_videos'] += 1
                    processed_demos.add(result['demo_name'])
                    
                    fps_info = f", FPS({result['fps_num_points']})"
                    tqdm.write(f"  [SUCCESS] {result['demo_name']}/{result['video_name']}: "
                              f"{result['total_frames']} frames, "
                              f"{result['valid_frames']} valid, "
                              f"{result['file_size']:.1f} KB{fps_info}")
                else:
                    tqdm.write(f"  [ERROR] {result['demo_name']}/{result['video_name']}: {result['error']}")
                
            except Exception as e:
                tqdm.write(f"  [ERROR] {demo_dir.name}/{video_name}: {str(e)}")
            
            # Update progress bar
            main_pbar.update(1)
            main_pbar.set_postfix({
                'demos': len(processed_demos),
                'success': stats['successful_videos']
            })
    
    main_pbar.close()
    stats['processed_demos'] = len(processed_demos)
    stats['total_output_files'] = len(all_output_files)
    
    print("\n" + "=" * 80)
    print("Processing Summary")
    print("=" * 80)
    print(f"Total demos: {stats['total_demos']}")
    print(f"Successfully processed demos: {stats['processed_demos']}")
    print(f"Total tactile videos: {stats['total_videos']}")
    print(f"Successfully processed videos: {stats['successful_videos']}")
    print(f"Generated point cloud files: {stats['total_output_files']}")
    
    if stats['total_videos'] > 0:
        success_rate = stats['successful_videos'] / stats['total_videos'] * 100
        print(f"Video processing success rate: {success_rate:.1f}%")
    
    if all_output_files:
        print(f"\nSuccessfully generated {len(all_output_files)} point cloud files:")
        total_size = 0
        for output_file in sorted(all_output_files):
            file_size = pathlib.Path(output_file).stat().st_size / 1024
            total_size += file_size
            print(f"  {output_file} ({file_size:.1f} KB)")
        
        print(f"\nTotal file size: {total_size:.1f} KB ({total_size/1024:.1f} MB)")
    
    print("=" * 80)


if __name__ == "__main__":
    # Multiprocessing safety: ensure proper process spawning
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        description="Extract point cloud data from tactile videos (Multiprocessing Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
1. Ensure you have recorded demo data with tactile videos
2. Implement your tactile point detection algorithm in the get_point function
3. Run this script to extract point cloud data from all tactile videos

Performance Optimizations:
- Uses multiprocessing instead of threading for true parallelism
- Automatically detects and utilizes all available CPU cores
- Bypasses Python GIL limitations for CPU-intensive image processing

Output:
- Each tactile video generates a corresponding *_points.npy file
- Files are saved in tactile_points/ subdirectory under each demo directory
- Each npy file contains point coordinate data for all frames (fixed shape: num_frames × fps_num_points × 3)

Example:
python 06_get_tac_point.py --cfg config/task_config.yaml
        """)
    parser.add_argument('--cfg', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.cfg)