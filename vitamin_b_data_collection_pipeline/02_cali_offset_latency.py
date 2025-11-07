#!/usr/bin/env python3

import os
import sys
import json
import csv
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from omegaconf import OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)

def convert_omegaconf_to_dict(config):
    if hasattr(config, '_content'):
        return OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, dict):
        return {k: convert_omegaconf_to_dict(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [convert_omegaconf_to_dict(item) for item in config]
    else:
        return config

def setup_logging(task_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = task_dir / f"latency_calibration_detailed_{timestamp}.log"
    
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
    logger.info("LATENCY CALIBRATION - DETAILED LOG")
    logger.info("="*80)
    
    return log_file

def log_and_print(message: str, level: str = "info", print_to_console: bool = False):
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

def load_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    return cfg

def get_latency_for_camera_type(camera_name: str, latencies: Dict) -> float:
    visual_cam_latency = latencies.get("visual_cam_latency", 0)
    tactile_cam_latency = latencies.get("tactile_cam_latency", 0)
    pose_latency = latencies.get("pose_latency", 0)
    
    if "visual" in camera_name.lower():
        logger.debug(f"Camera {camera_name} identified as visual, latency: {visual_cam_latency}s")
        return visual_cam_latency
    elif "tactile" in camera_name.lower():
        logger.debug(f"Camera {camera_name} identified as tactile, latency: {tactile_cam_latency}s")
        return tactile_cam_latency
    else:
        logger.debug(f"Camera {camera_name} identified as pose/other, latency: {pose_latency}s")
        return pose_latency

def process_video_timestamp_csv(csv_path: Path, latency: float) -> int:
    logger.info(f"Processing video timestamp: {csv_path.name} (latency: {latency}s)")
    
    try:
        rows = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                original_timestamp = float(row['timestamp'])
                adjusted_timestamp = original_timestamp - latency
                row['timestamp'] = adjusted_timestamp
                rows.append(row)
        
        logger.debug(f"Read {len(rows)} timestamp entries from {csv_path.name}")
        
        with open(csv_path, 'w', newline='') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        
        logger.info(f"[SUCCESS] Updated {len(rows)} timestamps in {csv_path.name}")
        return len(rows)
        
    except Exception as e:
        logger.error(f"Error processing {csv_path.name}: {e}")
        return 0

def process_trajectory_json(json_path: Path, latency: float) -> int:
    logger.info(f"Processing trajectory data: {json_path.name} (latency: {latency}s)")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.warning(f"Unexpected data format in {json_path.name}, skipping")
            return 0
        
        logger.debug(f"Read {len(data)} pose entries from {json_path.name}")
        
        processed_count = 0
        for pose in data:
            if 'timestamp_unix' in pose:
                original_timestamp = pose['timestamp_unix']
                pose['timestamp_unix'] = original_timestamp - latency
                processed_count += 1
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"[SUCCESS] Updated {processed_count} timestamps in {json_path.name}")
        return processed_count
        
    except Exception as e:
        logger.error(f"Error processing {json_path.name}: {e}")
        return 0

def process_single_demo(demo_dir: Path, latencies: Dict) -> Dict[str, int]:
    logger.info(f"Processing demo: {demo_dir.name}")
    
    results = {
        'csv_files_processed': 0,
        'csv_timestamps_updated': 0,
        'errors': 0
    }
    
    videos_dir = demo_dir / "videos"
    if videos_dir.exists():
        csv_files = list(videos_dir.glob("*_timestamps.csv"))
        logger.info(f"Found {len(csv_files)} timestamp CSV files in {demo_dir.name}")
        
        for csv_file in tqdm(csv_files, desc=f"Processing {demo_dir.name}", leave=False):
            camera_name = csv_file.stem.replace("_timestamps", "")
            latency = get_latency_for_camera_type(camera_name, latencies)
            
            try:
                updated_count = process_video_timestamp_csv(csv_file, latency)
                results['csv_files_processed'] += 1
                results['csv_timestamps_updated'] += updated_count
            except Exception as e:
                logger.error(f"Failed to process {csv_file}: {e}")
                results['errors'] += 1
    else:
        logger.warning(f"Videos directory not found in {demo_dir.name}")
    
    logger.info(f"Demo {demo_dir.name} processed: {results}")
    return results

def process_trajectory_data(trajectory_dir: Path, latencies: Dict) -> Dict[str, int]:
    logger.info(f"Processing trajectory data directory: {trajectory_dir}")
    
    results = {
        'json_files_processed': 0,
        'json_timestamps_updated': 0,
        'errors': 0
    }
    
    if not trajectory_dir.exists():
        logger.warning(f"Trajectory directory not found: {trajectory_dir}")
        return results
    
    json_files = list(trajectory_dir.glob("*.json"))
    # Filter out session_summary files
    json_files = [f for f in json_files if not f.name.startswith("session_summary")]
    logger.info(f"Found {len(json_files)} trajectory JSON files (excluding session_summary)")
    
    for json_file in tqdm(json_files, desc="Processing trajectory files"):
        try:
            pose_latency = latencies.get("pose_latency", 0)
            updated_count = process_trajectory_json(json_file, pose_latency)
            results['json_files_processed'] += 1
            results['json_timestamps_updated'] += updated_count
        except Exception as e:
            logger.error(f"Failed to process {json_file}: {e}")
            results['errors'] += 1
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Apply latency compensation to video timestamps and trajectory data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
    python 02_cali_offset_latency.py --cfg config/task_config.yaml
        """)
    
    parser.add_argument('--cfg', type=str, required=True,
                       help='Configuration file path')
    parser.add_argument('--dry_run', action='store_true',
                       help='Dry run mode, do not modify files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.cfg):
        print(f"[ERROR] Configuration file not found: {args.cfg}")
        sys.exit(1)
    
    try:
        cfg = load_config(args.cfg)
        task_name = cfg["task"]["name"]
        latencies = cfg["output_train_data"]
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        sys.exit(1)
    
    task_dir = Path("data") / task_name
    if not task_dir.exists():
        print(f"[ERROR] Task directory not found: {task_dir}")
        sys.exit(1)
    
    log_file = setup_logging(task_dir)
    print(f"[INFO] Detailed log: {log_file.name}")
    
    logger.info(f"Configuration file: {args.cfg}")
    logger.info(f"Task name: {task_name}")
    logger.info(f"Task directory: {task_dir}")
    logger.info(f"Latency configuration: {convert_omegaconf_to_dict(latencies)}")
    
    print(f"Task: {task_name}")
    print(f"  Latency config:")
    print(f"   Visual camera latency: {latencies.get('visual_cam_latency', 0)}s")
    print(f"    Tactile camera latency: {latencies.get('tactile_cam_latency', 0)}s")
    print(f"   Pose latency: {latencies.get('pose_latency', 0)}s")
    
    if args.dry_run:
        print(" DRY RUN MODE - No files will be modified")
        return
    
    try:
        total_stats = {
            'demos_processed': 0,
            'csv_files_processed': 0,
            'csv_timestamps_updated': 0,
            'json_files_processed': 0,
            'json_timestamps_updated': 0
        }
        
        demos_dir = task_dir / "demos"
        if demos_dir.exists():
            demo_dirs = [d for d in demos_dir.iterdir() if d.is_dir() and d.name.startswith('demo_')]
            logger.info(f"Found {len(demo_dirs)} demo directories")
            
            for demo_dir in tqdm(demo_dirs, desc="Processing demos", unit="demo"):
                demo_results = process_single_demo(demo_dir, latencies)
                total_stats['demos_processed'] += 1
                total_stats['csv_files_processed'] += demo_results['csv_files_processed']
                total_stats['csv_timestamps_updated'] += demo_results['csv_timestamps_updated']
        else:
            error_msg = f"Demos directory not found: {demos_dir}"
            logger.error(error_msg)
            print(f"[ERROR] {error_msg}")
        
        trajectory_dir = task_dir / "all_trajectory"
        if trajectory_dir.exists():
            trajectory_results = process_trajectory_data(trajectory_dir, latencies)
            total_stats['json_files_processed'] = trajectory_results['json_files_processed']
            total_stats['json_timestamps_updated'] = trajectory_results['json_timestamps_updated']
        else:
            error_msg = f"Trajectory directory not found: {trajectory_dir}"
            logger.warning(error_msg)
            print(f"[WARNING]  {error_msg}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = {
            'task_name': task_name,
            'processing_time': datetime.now().isoformat(),
            'log_file': str(log_file),
            'latency_configuration': convert_omegaconf_to_dict(latencies),
            'processing_statistics': total_stats
        }
        
        summary_file = task_dir / f"latency_calibration_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n{'='*60}")
        print(f" LATENCY CALIBRATION SUMMARY")
        print(f"{'='*60}")
        print(f"[SUCCESS] Demos processed: {total_stats['demos_processed']}")
        print(f"CSV files processed: {total_stats['csv_files_processed']}")
        print(f"CSV timestamps updated: {total_stats['csv_timestamps_updated']}")
        print(f"JSON files processed: {total_stats['json_files_processed']}")
        print(f"JSON timestamps updated: {total_stats['json_timestamps_updated']}")
        print(f"[INFO] Summary saved: {summary_file.name}")
        print(f"[INFO] Detailed log: {log_file.name}")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"[ERROR] Error occurred during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()