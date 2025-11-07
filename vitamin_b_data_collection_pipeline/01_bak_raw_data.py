#!/usr/bin/env python3

import os
import sys
import shutil
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

def load_config(config_path: str) -> OmegaConf:
    try:
        config = OmegaConf.load(config_path)
        print(f"[CONFIG] Successfully loaded config: {config_path}")
        return config
    except Exception as e:
        print(f"[ERROR] Failed to load config '{config_path}': {e}")
        raise

def get_raw_data_folder_path(config: OmegaConf) -> str:
    if not hasattr(config, 'recorder') or not hasattr(config.recorder, 'output'):
        raise ValueError("[CONFIG] Missing required config item 'recorder.output'")
    if not hasattr(config, 'task') or not hasattr(config.task, 'name'):
        raise ValueError("[CONFIG] Missing required config item 'task.name'")
    
    base_dir = config.recorder.output
    task_name = config.task.name
    raw_data_folder = os.path.join(base_dir, task_name)
    
    return raw_data_folder

def format_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f}{size_names[i]}"

def get_folder_size(folder_path: str) -> int:
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    continue
    except (OSError, FileNotFoundError):
        pass
    return total_size

def copy_folder_with_progress(src: str, dst: str) -> None:
    print(f"[COPY] Starting folder copy...")
    print(f"[COPY]    Source: {src}")
    print(f"[COPY]    Destination: {dst}")
    
    total_size = get_folder_size(src)
    print(f"[COPY]    Folder size: {format_size(total_size)}")
    
    if total_size == 0:
        print(f"[WARNING]  Source folder is empty or inaccessible")
    
    try:
        shutil.copytree(src, dst, dirs_exist_ok=False)
        
        copied_size = get_folder_size(dst)
        print(f"[COPY] Copy completed!")
        print(f"[COPY]    Copied size: {format_size(copied_size)}")
        
        if copied_size != total_size and total_size > 0:
            print(f"[WARNING]  Copied size doesn't match source size, please check")
        
    except FileExistsError:
        print(f"[ERROR]  Destination folder already exists: {dst}")
        raise
    except PermissionError:
        print(f"[ERROR]  Permission denied, cannot copy to: {dst}")
        raise
    except Exception as e:
        print(f"[ERROR]  Error during copy: {e}")
        raise

def backup_raw_data(config_path: str, custom_suffix: str = None) -> None:
    print("ViTaMIn-B Raw Data Backup Tool")
    print("=" * 60)
    
    config = load_config(config_path)
    
    copy_raw_data = getattr(config.task, 'copy_raw_data', True)
    print(f"[CONFIG] copy_raw_data = {copy_raw_data}")
    
    if not copy_raw_data:
        print(f"[SKIP] copy_raw_data = False, skipping backup")
        print("=" * 60)
        return True
    
    print(f"[INFO] copy_raw_data = True, starting backup")
    
    raw_data_folder = get_raw_data_folder_path(config)
    print(f"[DETECT] Detected raw data folder: {raw_data_folder}")
    
    if not os.path.exists(raw_data_folder):
        print(f"[ERROR]  Raw data folder does not exist: {raw_data_folder}")
        print(f"[HINT] Please check path settings in config file")
        return False
    
    if not os.path.isdir(raw_data_folder):
        print(f"[ERROR]  Path is not a directory: {raw_data_folder}")
        return False
    
    if custom_suffix:
        backup_suffix = custom_suffix
    else:
        backup_suffix = "bak"
    
    backup_folder = f"{raw_data_folder}_{backup_suffix}"
    
    if os.path.exists(backup_folder):
        print(f"[ERROR]  Backup folder already exists: {backup_folder}")
        user_input = input("[PROMPT] Delete existing backup and recreate? (y/N): ").strip().lower()
        if user_input == 'y':
            print(f"[CLEAN]  Deleting existing backup folder...")
            shutil.rmtree(backup_folder)
            print(f"[CLEAN] Existing backup deleted")
        else:
            print(f"[ABORT]  Backup operation cancelled")
            return False
    
    try:
        copy_folder_with_progress(raw_data_folder, backup_folder)
        
        print("\n" + "=" * 60)
        print("Backup completed!")
        print(f"[SUCCESS] Raw data: {raw_data_folder}")
        print(f"[SUCCESS] Backup location: {backup_folder}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR]  Backup failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="ViTaMIn-B Raw Data Backup Tool")
    parser.add_argument("--cfg", 
                       default="/mnt/disk_1_4T/Chuanyu/codehub/ViTaMIn-B/config/task_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--suffix", 
                       default="bak",
                       help="Backup folder suffix (default: bak)")
    
    args = parser.parse_args()
    
    try:
        success = backup_raw_data(args.cfg, args.suffix)
        
        if success:
            print(f"\n[INFO] Backup operation completed successfully!")
            sys.exit(0)
        else:
            print(f"\n[INFO]  Backup operation not completed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n[ABORT]  Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR]  Program execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()