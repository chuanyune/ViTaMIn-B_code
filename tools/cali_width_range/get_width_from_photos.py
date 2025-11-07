import sys
import os
from pathlib import Path
import argparse
import json
import yaml
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)

from utils.cv_util import (
    parse_aruco_config, 
    parse_fisheye_intrinsics,
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    get_gripper_width
)


def get_sorted_image_files(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder does not exist: {folder_path}")
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    image_files.sort(key=lambda x: x.name)
    
    return image_files


def detect_aruco_in_image(img, aruco_dict, marker_size_map, fisheye_intr):
    tag_dict = detect_localize_aruco_tags(
        img=img,
        aruco_dict=aruco_dict,
        marker_size_map=marker_size_map,
        fisheye_intr_dict=fisheye_intr,
        refine_subpix=True
    )
    return tag_dict


def calculate_widths_from_photos(folder_path, config_file):
    cfg = OmegaConf.load(config_file)
    
    use_gelsight = cfg["calculate_width"]["use_gelsight"]
    if use_gelsight:
        aruco_yaml = cfg["calculate_width"]["aruco_yaml_gelsight"]
    else:
        aruco_yaml = cfg["calculate_width"]["aruco_yaml_finray"]
    
    camera_intrinsics = cfg["calculate_width"]["cam_intrinsic_json_path"]
    left_hand_left_id = cfg["calculate_width"]["left_hand_aruco_id"]["left_id"]
    left_hand_right_id = cfg["calculate_width"]["left_hand_aruco_id"]["right_id"]
    right_hand_left_id = cfg["calculate_width"]["right_hand_aruco_id"]["left_id"]
    right_hand_right_id = cfg["calculate_width"]["right_hand_aruco_id"]["right_id"]
    
    print(f"Using ArUco config: {aruco_yaml}")
    print(f"Using camera intrinsics: {camera_intrinsics}")
    print(f"Left hand ArUco IDs: {left_hand_left_id}, {left_hand_right_id}")
    print(f"Right hand ArUco IDs: {right_hand_left_id}, {right_hand_right_id}")
    
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, 'r')))
    aruco_dict = aruco_config['aruco_dict']
    marker_size_map = aruco_config['marker_size_map']
    
    raw_fisheye_intr = parse_fisheye_intrinsics(json.load(open(camera_intrinsics, 'r')))
    
    image_files = get_sorted_image_files(folder_path)
    if not image_files:
        print(f"No image files found in folder {folder_path}")
        return None
    
    print(f"Found {len(image_files)} images")
    print("="*60)
    
    results = []
    left_hand_detected = False
    right_hand_detected = False
    
    for i, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"Cannot read image: {image_file}")
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_res = np.array([img.shape[1], img.shape[0]])
            fisheye_intr = convert_fisheye_intrinsics_resolution(
                opencv_intr_dict=raw_fisheye_intr, 
                target_resolution=img_res
            )
            
            tag_dict = detect_aruco_in_image(img_rgb, aruco_dict, marker_size_map, fisheye_intr)
            
            left_width = get_gripper_width(tag_dict, left_hand_left_id, left_hand_right_id)
            if left_width is not None:
                left_hand_detected = True
            
            right_width = get_gripper_width(tag_dict, right_hand_left_id, right_hand_right_id)
            if right_width is not None:
                right_hand_detected = True
            
            result = {
                'image_index': i,
                'image_name': image_file.name,
                'left_hand_width': left_width,
                'right_hand_width': right_width,
                'detected_tags': list(tag_dict.keys())
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing image {image_file}: {str(e)}")
            continue
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("Detection results statistics:")
    print(f"Total images: {len(image_files)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Left hand detected: {left_hand_detected}")
    print(f"Right hand detected: {right_hand_detected}")
    
    if left_hand_detected:
        left_valid = df['left_hand_width'].notna().sum()
        print(f"Left hand valid width measurements: {left_valid}/{len(results)}")
        if left_valid > 0:
            print(f"Left hand width range: {df['left_hand_width'].min():.4f} - {df['left_hand_width'].max():.4f} meters")
    
    if right_hand_detected:
        right_valid = df['right_hand_width'].notna().sum()
        print(f"Right hand valid width measurements: {right_valid}/{len(results)}")
        if right_valid > 0:
            print(f"Right hand width range: {df['right_hand_width'].min():.4f} - {df['right_hand_width'].max():.4f} meters")
    
    return df


def save_results(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect ArUco markers from photo folder and calculate gripper width",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python get_width_from_photos.py --folder photos --config config/data_collection.yaml
  python get_width_from_photos.py --folder photos --config config/data_collection.yaml --output results.csv
        """)
    
    parser.add_argument('--folder', type=str, required=True,
                       help='Folder path containing photos')
    parser.add_argument('--config', type=str, default='config/data_collection.yaml',
                       help='Configuration file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"Error: Folder does not exist {args.folder}")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file does not exist {args.config}")
        return
    
    results_df = calculate_widths_from_photos(args.folder, args.config)
    
    if results_df is not None:
        print("\nFirst few rows of results:")
        print(results_df.head())
        
        if args.output:
            save_results(results_df, args.output)
        else:
            folder_name = os.path.basename(args.folder)
            default_output = f"{folder_name}_width_results.csv"
            save_results(results_df, default_output)


if __name__ == "__main__":
    main()
