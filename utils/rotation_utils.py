"""
Rotation utilities module - for detecting and fixing jumps in quaternion rotation data
"""

import numpy as np
import pandas as pd
import os
import glob
from scipy.spatial.transform import Rotation, Slerp

def calculate_rotation_angle(rot1, rot2):
    """
    Calculate angular change between two rotations (in degrees)
    
    Args:
        rot1: First rotation representation (can be quaternion or axis-angle)
        rot2: Second rotation representation (can be quaternion or axis-angle)
        
    Returns:
        Angular difference between two rotations (degrees)
    """
    # Determine how to create rotation object based on input type
    if isinstance(rot1, Rotation):
        rotation1 = rot1
    elif len(rot1) == 3:  # Axis-angle representation
        rotation1 = Rotation.from_rotvec(np.array(rot1))
    elif len(rot1) == 4:  # Quaternion representation [x, y, z, w]
        rotation1 = Rotation.from_quat(rot1)
    else:
        raise ValueError("Unrecognized rotation representation format")
        
    if isinstance(rot2, Rotation):
        rotation2 = rot2
    elif len(rot2) == 3:  # Axis-angle representation
        rotation2 = Rotation.from_rotvec(np.array(rot2))
    elif len(rot2) == 4:  # Quaternion representation [x, y, z, w]
        rotation2 = Rotation.from_quat(rot2)
    else:
        raise ValueError("Unrecognized rotation representation format")
    
    # Calculate relative rotation
    relative_rotation = rotation1.inv() * rotation2
    
    # Get rotation angle (in radians)
    angle_radians = relative_rotation.magnitude()
    
    # Convert to degrees
    angle_degrees = np.abs(angle_radians * 180 / np.pi)
    
    return angle_degrees

def detect_rotation_jumps(rotations, threshold_degrees=10):
    """
    Detect jump positions in rotation sequence
    
    Args:
        rotations: List of rotation objects, quaternion array, or axis-angle array
        threshold_degrees: Angular jump threshold (degrees)
        
    Returns:
        List of indices where jumps are detected
    """
    jump_indices = []
    
    for i in range(1, len(rotations)):
        angle = calculate_rotation_angle(rotations[i-1], rotations[i])
        if angle > threshold_degrees:
            jump_indices.append(i)
            
    return jump_indices

def detect_and_fix_rotation_jumps(df, threshold_degrees=10, verbose=True):
    """
    Detect large rotation jumps (greater than threshold_degrees) and fix using interpolation
    
    Args:
        df: DataFrame containing quaternion columns (q_x, q_y, q_z, q_w) and timestamp
        threshold_degrees: Angular threshold for detecting rotation jumps (degrees)
        verbose: Whether to print detailed information
        
    Returns:
        Fixed DataFrame
    """
    # Create copy to avoid modifying original data
    fixed_df = df.copy()
    
    # Get quaternions
    quats = fixed_df[['q_x', 'q_y', 'q_z', 'q_w']].values
    quats_normalized = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    
    # Convert to rotation objects
    rot_objects = Rotation.from_quat(quats_normalized)
    
    # Create frame validity mask (frames that don't need interpolation are marked as True)
    valid_mask = np.ones(len(fixed_df), dtype=bool)
    
    # Jump counter
    jump_count = 0
    
    # Check rotation jumps between consecutive frames
    for i in range(1, len(rot_objects)):
        # Calculate relative rotation needed to rotate from previous frame to current frame
        relative_rotation = rot_objects[i-1].inv() * rot_objects[i]
        
        # Get rotation angle (in radians)
        angle_radians = relative_rotation.magnitude()
        
        # Convert to degrees
        angle_degrees = np.abs(angle_radians * 180 / np.pi)
        
        # If angle difference is greater than threshold, mark as invalid frame
        if angle_degrees > threshold_degrees:
            valid_mask[i] = False
            jump_count += 1
            if verbose:
                print(f"Detected rotation jump: {angle_degrees:.2f}° (index {i}, timestamp {fixed_df['timestamp'].iloc[i]:.3f}s)")
    
    # If jumps detected
    if jump_count > 0:
        if verbose:
            print(f"Found {jump_count} rotation jumps greater than {threshold_degrees}°, fixing using interpolation...")
        
        # Get indices of valid frames
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < 2:
            if verbose:
                print("Warning: Not enough valid frames for interpolation, skipping rotation fix")
            return df
        
        # Get timestamps and rotations of valid frames
        valid_times = fixed_df['timestamp'].iloc[valid_indices].values
        valid_rotations = rot_objects[valid_indices]
        
        # Create spherical linear interpolator
        slerp = Slerp(valid_times, valid_rotations)
        
        # Get all timestamps
        all_times = fixed_df['timestamp'].values
        
        # Interpolate rotations for all frames
        interpolated_quats = slerp(all_times).as_quat()
        
        # Update quaternions in DataFrame
        fixed_df['q_x'] = interpolated_quats[:, 0]
        fixed_df['q_y'] = interpolated_quats[:, 1]
        fixed_df['q_z'] = interpolated_quats[:, 2]
        fixed_df['q_w'] = interpolated_quats[:, 3]
        
        if verbose:
            print(f"[Success] Fixed {jump_count} rotation jumps using interpolation")
    else:
        if verbose:
            print("No significant rotation jumps detected")
    
    return fixed_df

def analyze_rotation_angles(df):
    """
    Analyze rotation angle changes in data frame
    
    Args:
        df: DataFrame containing quaternion columns
        
    Returns:
        Dictionary containing statistics like max angle, mean angle, standard deviation, etc.
    """
    # Get quaternions
    quats = df[['q_x', 'q_y', 'q_z', 'q_w']].values
    quats_normalized = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    
    # Convert to rotation objects
    rot_objects = Rotation.from_quat(quats_normalized)
    
    # Calculate angle differences between adjacent frames
    angles = []
    max_angle_idx = 0
    max_angle = 0
    
    for i in range(1, len(rot_objects)):
        relative_rotation = rot_objects[i-1].inv() * rot_objects[i]
        angle_degrees = np.abs(relative_rotation.magnitude() * 180 / np.pi)
        angles.append(angle_degrees)
        
        if angle_degrees > max_angle:
            max_angle = angle_degrees
            max_angle_idx = i
    
    # Calculate statistics
    angles = np.array(angles)
    stats = {
        'max_angle': max_angle,
        'max_angle_idx': max_angle_idx,
        'mean_angle': np.mean(angles),
        'median_angle': np.median(angles),
        'std_angle': np.std(angles),
        'total_frames': len(df),
        'total_angle_changes': len(angles)
    }
    
    return stats

def visualize_rotation_jumps(df, threshold_degrees=10):
    """
    Visualize rotation jumps
    
    Args:
        df: DataFrame containing quaternion columns
        threshold_degrees: Angular threshold for detecting rotation jumps (degrees)
    """
    try:
        import matplotlib.pyplot as plt
        
        # Get quaternions
        quats = df[['q_x', 'q_y', 'q_z', 'q_w']].values
        quats_normalized = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        
        # Convert to rotation objects
        rot_objects = Rotation.from_quat(quats_normalized)
        
        # Calculate angle differences between adjacent frames
        angles = []
        timestamps = df['timestamp'].values
        time_diffs = []
        
        for i in range(1, len(rot_objects)):
            relative_rotation = rot_objects[i-1].inv() * rot_objects[i]
            angle_degrees = np.abs(relative_rotation.magnitude() * 180 / np.pi)
            angles.append(angle_degrees)
            time_diffs.append(timestamps[i] - timestamps[i-1])
        
        # Plot angle differences
        plt.figure(figsize=(12, 6))
        
        # Main plot: Angle differences
        plt.subplot(2, 1, 1)
        plt.plot(angles, 'b-', label='Rotation Angle Changes')
        plt.axhline(y=threshold_degrees, color='r', linestyle='--', label=f'Threshold ({threshold_degrees}°)')
        plt.xlabel('Frame Index')
        plt.ylabel('Angle Difference (°)')
        plt.title('Rotation Angle Changes Between Adjacent Frames')
        plt.legend()
        plt.grid(True)
        
        # Highlight points exceeding threshold
        jump_indices = [i for i, angle in enumerate(angles) if angle > threshold_degrees]
        if jump_indices:
            plt.scatter([jump_indices], [angles[i] for i in jump_indices], color='red', s=50, 
                       label=f'Jumps ({len(jump_indices)})')
            plt.legend()
        
        # Subplot: Time differences
        plt.subplot(2, 1, 2)
        plt.plot(time_diffs, 'g-', label='Time Intervals')
        plt.xlabel('Frame Index')
        plt.ylabel('Time Difference (s)')
        plt.title('Time Intervals Between Adjacent Frames')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return angles, jump_indices
    
    except ImportError:
        print("Cannot import matplotlib, skipping visualization")
        return None, None

def process_folder(folder_path, file_pattern='camera_trajectory.csv', threshold_degrees=10, fix_jumps=False, output_folder=None, visualize=False):
    """
    Process all matching CSV files in folder, detect rotation jumps
    
    Args:
        folder_path: Folder path
        file_pattern: Filename matching pattern (defaults to 'camera_trajectory.csv')
        threshold_degrees: Angular jump threshold (degrees)
        fix_jumps: Whether to fix detected jumps
        output_folder: Output folder for fixed files (if None, don't save)
        visualize: Whether to visualize rotation angles for each file
        
    Returns:
        Dictionary containing statistics for each file
    """
    # Find all matching files
    search_pattern = os.path.join(folder_path, '**', file_pattern)
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"No files matching {file_pattern} found in {folder_path}")
        return {}
    
    print(f"Found {len(files)} matching files")
    
    # Prepare results dictionary and statistics data
    results = {}
    stats_summary = []
    
    # Create output folder
    if fix_jumps and output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Process each file
    for file_path in files:
        rel_path = os.path.relpath(file_path, folder_path)
        print(f"\nProcessing file: {rel_path}")
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            required_columns = ['timestamp', 'q_x', 'q_y', 'q_z', 'q_w']
            if not all(col in df.columns for col in required_columns):
                print(f"  Skip: Missing required columns {required_columns}")
                continue
            
            # Analyze rotation angles
            stats = analyze_rotation_angles(df)
            print(f"  Data frames: {stats['total_frames']}")
            print(f"  Max rotation angle: {stats['max_angle']:.2f}° (index {stats['max_angle_idx']})")
            print(f"  Average rotation angle: {stats['mean_angle']:.2f}°")
            
            # Add file path to statistics
            stats['file_path'] = file_path
            stats['relative_path'] = rel_path
            stats_summary.append(stats)
            
            # Detect rotation jumps
            jump_indices = detect_rotation_jumps(df[['q_x', 'q_y', 'q_z', 'q_w']].values, threshold_degrees)
            
            if jump_indices:
                print(f"  Detected {len(jump_indices)} rotation jumps (threshold {threshold_degrees}°)")
                for idx in jump_indices:
                    if 'timestamp' in df.columns:
                        print(f"    - Index {idx}, timestamp {df['timestamp'].iloc[idx]:.3f}s, angle {stats['angles'][idx-1]:.2f}°")
                    else:
                        print(f"    - Index {idx}, angle {stats['angles'][idx-1]:.2f}°")
                
                # Fix jumps
                if fix_jumps:
                    fixed_df = detect_and_fix_rotation_jumps(df, threshold_degrees, verbose=False)
                    print(f"  Fixed {len(jump_indices)} rotation jumps")
                    
                    # Save fixed file
                    if output_folder:
                        output_path = os.path.join(output_folder, rel_path)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        fixed_df.to_csv(output_path, index=False)
                        print(f"  Saved fixed file to: {output_path}")
            else:
                print(f"  No rotation jumps detected (threshold {threshold_degrees}°)")
            
            # Visualize
            if visualize:
                angles, _ = visualize_rotation_jumps(df, threshold_degrees)
                stats['angles'] = angles
            
            # Save results
            results[file_path] = {
                'stats': stats,
                'jump_indices': jump_indices,
                'has_jumps': len(jump_indices) > 0
            }
            
        except Exception as e:
            print(f"  Error processing file: {str(e)}")
    
    # Generate report
    if stats_summary:
        # Convert to DataFrame for easier analysis
        stats_df = pd.DataFrame(stats_summary)
        
        # Sort by max rotation angle
        stats_df_sorted = stats_df.sort_values('max_angle', ascending=False)
        
        # Output summary report
        print("\n=== Rotation Angle Analysis Summary ===")
        print(f"Total files: {len(stats_df)}")
        print(f"Max rotation angle: {stats_df['max_angle'].max():.2f}°")
        print(f"Average max rotation angle: {stats_df['max_angle'].mean():.2f}°")
        print(f"Files with max angle exceeding {threshold_degrees}°: {(stats_df['max_angle'] > threshold_degrees).sum()}")
        
        # Output top 10 files with largest rotation angles
        print("\nTop 10 files with largest rotation angles:")
        for idx, row in stats_df_sorted.head(10).iterrows():
            print(f"  {row['relative_path']}: {row['max_angle']:.2f}° (index {row['max_angle_idx']})")
        
        # Save report to CSV
        report_path = os.path.join(folder_path, "rotation_analysis_report.csv")
        stats_df_sorted.to_csv(report_path, index=False)
        print(f"\nDetailed report saved to: {report_path}")
    
    return results

if __name__ == "__main__":
    # Test code
    import argparse
    
    parser = argparse.ArgumentParser(description="Rotation jump detection and repair tool")
    parser.add_argument('--csv', type=str, help='Path to CSV file containing quaternions')
    parser.add_argument('--folder', type=str, default="E:\\codehub\\ViTaMIn-B\\Data_collection\\data\\vitaminb_single_420_2_humanoid_robot_pick_and_place\\demo_single", help='Folder path to process')
    parser.add_argument('--file_pattern', type=str, default='camera_trajectory_old.csv', help='Filename matching pattern')
    parser.add_argument('--threshold', type=float, default=10.0, help='Rotation jump angle threshold (degrees)')
    parser.add_argument('--fix', action='store_true', help='Whether to fix detected jumps')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize rotation jumps')
    parser.add_argument('--output', type=str, help='Output path for fixed files')
    
    args = parser.parse_args()
    
    if args.folder:
        # Process folder
        process_folder(
            args.folder, 
            file_pattern=args.file_pattern,
            threshold_degrees=args.threshold,
            fix_jumps=args.fix,
            output_folder=args.output,
            visualize=args.visualize
        )
    elif args.csv:
        # Process single CSV file
        try:
            df = pd.read_csv(args.csv)
            print(f"Successfully read CSV file: {args.csv}")
            print(f"Data frames: {len(df)}")
            
            # Analyze rotation angles
            stats = analyze_rotation_angles(df)
            print(f"Max rotation angle: {stats['max_angle']:.2f}° (index {stats['max_angle_idx']})")
            print(f"Average rotation angle: {stats['mean_angle']:.2f}°")
            
            # Detect and fix rotation jumps
            if args.fix:
                fixed_df = detect_and_fix_rotation_jumps(df, args.threshold)
                
                # Save fixed data
                if args.output:
                    fixed_df.to_csv(args.output, index=False)
                    print(f"Saved fixed data to: {args.output}")
            else:
                jump_indices = detect_rotation_jumps(df[['q_x', 'q_y', 'q_z', 'q_w']].values, args.threshold)
                if jump_indices:
                    print(f"Detected {len(jump_indices)} rotation jumps (threshold {args.threshold}°)")
                    for idx in jump_indices:
                        if 'timestamp' in df.columns:
                            print(f"  - Index {idx}, timestamp {df['timestamp'].iloc[idx]:.3f}s")
                        else:
                            print(f"  - Index {idx}")
                else:
                    print(f"No rotation jumps detected (threshold {args.threshold}°)")
            
            # Visualize
            if args.visualize:
                visualize_rotation_jumps(df, args.threshold)
            
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Please provide --csv or --folder parameter")
        parser.print_help() 