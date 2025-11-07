# %%
import sys
import os
from pathlib import Path
project_root = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(project_root))

import pickle
import argparse
from omegaconf import OmegaConf
from utils.cv_util import get_gripper_width
import concurrent.futures
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import pathlib


def interpolate_missing_widths(width_df: pd.DataFrame):
    if 'frame' not in width_df.columns or 'width' not in width_df.columns:
        print("Warning: Width CSV file missing required columns (frame, width)")
        return width_df
    
    df = width_df.copy()
    df['is_valid'] = (~df['width'].isna()) & (df['width'] > 0)
    valid_frames = df.loc[df['is_valid'] == True]
    
    if len(valid_frames) < 2:
        print(f"Warning: Only {len(valid_frames)} valid points found, need at least 2 for interpolation")
        if len(valid_frames) == 1:
            fill_value = valid_frames['width'].iloc[0]
            df.loc[df['is_valid'] != True, 'width'] = fill_value
            print(f"Filled {len(df) - len(valid_frames)} frames with constant value: {fill_value:.4f}")
        elif len(valid_frames) == 0:
            default_width = 0.05
            df['width'] = default_width
            df['is_valid'] = False
            print(f"No valid width measurements found, filled all {len(df)} frames with default value: {default_width}")
        return df
    
    try:
        interpolator = interp1d(
            valid_frames['frame'].values, 
            valid_frames['width'].values,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        invalid_mask = df['is_valid'] != True
        invalid_frames = df.loc[invalid_mask]
        
        if len(invalid_frames) > 0:
            interpolated_values = interpolator(invalid_frames['frame'].values)
            df.loc[invalid_mask, 'width'] = interpolated_values
            print(f"Successfully interpolated {len(invalid_frames)} frames with invalid width measurements")
        else:
            print("All frames already have valid width measurements, no interpolation needed")
            
        if df['width'].isna().any():
            print("Warning: Still have NaN values after interpolation, filling with nearest valid value")
            df['width'] = df['width'].ffill().bfill()
            
    except Exception as e:
        print(f"Width interpolation failed: {str(e)}")
        df['width'] = df['width'].ffill().bfill()
        if df['width'].isna().any():
            default_width = 0.05
            df['width'] = df['width'].fillna(default_width)
            print(f"Used fallback strategy to fill remaining NaN values with {default_width}")
        
    return df


def process_pkl_file(pkl_file, left_id, right_id, nominal_z, output_path):
    tag_detection_results = pickle.load(open(pkl_file, 'rb'))
    gripper_widths = []
    valid_count = 0
    total_frames = len(tag_detection_results)
    
    for i, dt in enumerate(tag_detection_results):
        tag_dict = dt['tag_dict']
        width = get_gripper_width(tag_dict, left_id, right_id, nominal_z=nominal_z)
        
        if width is not None and not pd.isna(width) and width > 0:
            valid_count += 1
            
        gripper_widths.append({
            'frame': i,
            'width': width
        })
    
    print(f"Processing {pkl_file}: {valid_count}/{total_frames} frames have valid width detection")
    
    df = pd.DataFrame(gripper_widths)
    df = interpolate_missing_widths(df)
    
    if df['width'].isna().any():
        print(f"Warning: Still have NaN values in {output_path}, applying final cleanup")
        df['width'] = df['width'].fillna(0.05)
    
    if 'is_valid' not in df.columns:
        df['is_valid'] = (~df['width'].isna()) & (df['width'] > 0)
    df['is_valid'] = df['is_valid'].fillna(False)
    
    nan_count = df['width'].isna().sum()
    if nan_count > 0:
        print(f"ERROR: Still have {nan_count} NaN values before saving to {output_path}")
        df['width'] = df['width'].fillna(0.05)
    
    print(f"Final validation: {len(df)} frames, {(~df['width'].isna()).sum()} non-null values")
    
    df.to_csv(output_path, index=False)
    
    return output_path


def main(config_file):
    cfg = OmegaConf.load(config_file)
    task_name = cfg["task"]["name"]
    task_type = cfg["task"]["type"]
    nominal_z = cfg["calculate_width"]["nominal_z"]
    left_hand_left_id = cfg["calculate_width"]["left_hand_aruco_id"]["left_id"]
    left_hand_right_id = cfg["calculate_width"]["left_hand_aruco_id"]["right_id"]
    right_hand_left_id = cfg["calculate_width"]["right_hand_aruco_id"]["left_id"]
    right_hand_right_id = cfg["calculate_width"]["right_hand_aruco_id"]["right_id"]
    folder = os.path.join("data", task_name, "demos")
    pkl_files = []
    
    for root, dirs, files in os.walk(folder):
        if os.path.basename(root).startswith('demo_'):
            for file in files:
                if file.lower().endswith(('.pkl')):
                    pkl_files.append(os.path.join(root, file))
    
    if task_type == "single":
        left_hand_pkl_files = []
        output_path_left = []
        for i, pkl_file in enumerate(pkl_files):
            pkl_filename = os.path.basename(pkl_file)
            if pkl_filename == 'tag_detection_left.pkl':
                left_hand_pkl_files.append(pkl_file)
                output_path_left.append(os.path.join(os.path.dirname(pkl_file), 'gripper_width_left.csv'))
        right_hand_pkl_files = []
        output_path_right = []
    elif task_type == "bimanual":
        left_hand_pkl_files = []
        output_path_left = []
        right_hand_pkl_files = []   
        output_path_right = []
        
        for i, pkl_file in enumerate(pkl_files):
            pkl_filename = os.path.basename(pkl_file)
            if pkl_filename == 'tag_detection_left.pkl':
                left_hand_pkl_files.append(pkl_file)
                output_path_left.append(os.path.join(os.path.dirname(pkl_file), 'gripper_width_left.csv'))
            elif pkl_filename == 'tag_detection_right.pkl':
                right_hand_pkl_files.append(pkl_file)
                output_path_right.append(os.path.join(os.path.dirname(pkl_file), 'gripper_width_right.csv'))
            else:
                print(f"Warning: Unknown pkl file format: {pkl_file}")
    else:
        raise ValueError(f"Unknown task type: {task_type}")
            

    print(f"Found {len(left_hand_pkl_files)} left hand files and {len(right_hand_pkl_files)} right hand files")
    
    saved_files = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        left_hand_futures = [
            executor.submit(process_pkl_file, pkl_file, left_hand_left_id, left_hand_right_id, nominal_z, output_path_left[i])
            for i, pkl_file in enumerate(left_hand_pkl_files)
        ]
        
        right_hand_futures = [
            executor.submit(process_pkl_file, pkl_file, right_hand_left_id, right_hand_right_id, nominal_z, output_path_right[i])
            for i, pkl_file in enumerate(right_hand_pkl_files)
        ]
        
        for future in concurrent.futures.as_completed(left_hand_futures + right_hand_futures):
            try:
                output_file = future.result()
                saved_files.append(output_file)
                print(f"Saved width data to: {output_file}")
            except Exception as e:
                print(f"Error processing file: {str(e)}")
    
    print("=" * 80)
    print("Gripper width calculation completed! Statistics:")
    print("=" * 80)
    
    print(f"Total pkl files: {len(pkl_files)}")
    print(f"Left hand pkl files: {len(left_hand_pkl_files)}")
    print(f"Right hand pkl files: {len(right_hand_pkl_files)}")
    print(f"[SUCCESS] Successfully generated CSV files: {len(saved_files)}")
    
    if len(saved_files) > 0:
        print(f"\nSuccessfully generated {len(saved_files)} CSV files:")
        
        total_frames = 0
        total_valid_frames = 0
        total_interpolated_frames = 0
        
        for csv_file in sorted(saved_files):
            try:
                df = pd.read_csv(csv_file)
                file_frames = len(df)
                valid_frames = len(df[~df['width'].isna() & (df['width'] > 0)])
                file_size = pathlib.Path(csv_file).stat().st_size / 1024
                
                print(f"   {csv_file}")
                print(f"    Total frames: {file_frames}")
                print(f"    [SUCCESS] Valid frames: {valid_frames}")
                print(f"     Valid rate: {valid_frames/file_frames*100:.1f}%" if file_frames > 0 else "     Valid rate: 0%")
                print(f"    File size: {file_size:.1f} KB")
                
                valid_widths = df[~df['width'].isna() & (df['width'] > 0)]['width']
                if len(valid_widths) > 0:
                    print(f"    Width range: {valid_widths.min():.4f} - {valid_widths.max():.4f}")
                    print(f"    Average width: {valid_widths.mean():.4f}")
                    print(f"    Width std: {valid_widths.std():.4f}")
                
                total_frames += file_frames
                total_valid_frames += valid_frames
                
                if 'is_valid' in df.columns:
                    interpolated_frames = len(df[df['is_valid'] == False])
                    total_interpolated_frames += interpolated_frames
                    if interpolated_frames > 0:
                        print(f"     Interpolated frames: {interpolated_frames}")
                
                print()
                
            except Exception as e:
                print(f"  [ERROR] Error reading {csv_file}: {str(e)}")
        
        print("Overall statistics:")
        print(f"  Total frames: {total_frames}")
        print(f"  [SUCCESS] Total valid frames: {total_valid_frames}")
        print(f"   Overall valid rate: {total_valid_frames/total_frames*100:.1f}%" if total_frames > 0 else "   Overall valid rate: 0%")
        if total_interpolated_frames > 0:
            print(f"   Total interpolated frames: {total_interpolated_frames}")
    
    expected_csv_files = []
    if task_type == "single":
        expected_csv_files = [os.path.join(os.path.dirname(pkl), 'gripper_width_left.csv') for pkl in left_hand_pkl_files]
    elif task_type == "bimanual":
        expected_csv_files = [os.path.join(os.path.dirname(pkl), 'gripper_width_left.csv') for pkl in left_hand_pkl_files]
        expected_csv_files.extend([os.path.join(os.path.dirname(pkl), 'gripper_width_right.csv') for pkl in right_hand_pkl_files])
    
    missing_csv_files = []
    for expected_csv in expected_csv_files:
        if expected_csv not in saved_files:
            missing_csv_files.append(expected_csv)
    
    if len(missing_csv_files) > 0:
        print(f"\n{len(missing_csv_files)} files not successfully processed:")
        for missing_csv in missing_csv_files:
            corresponding_pkl = missing_csv.replace('gripper_width_left.csv', 'tag_detection_left.pkl').replace('gripper_width_right.csv', 'tag_detection_right.pkl')
            print(f"  ✗ {missing_csv} (corresponding pkl: {corresponding_pkl})")
            
            if not os.path.exists(corresponding_pkl):
                print(f"    [ERROR] Corresponding pkl file does not exist: {corresponding_pkl}")
    
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Task name')
    args = parser.parse_args()
    config_file = args.cfg
    main(config_file)