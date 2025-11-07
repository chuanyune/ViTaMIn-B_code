# coding=utf-8

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Import configuration
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
from config.vitamin_b_config import *
from utils.quest_robot_module import QuestBimanualModule

CONFIG: Optional[OmegaConf] = None
CONFIG_FILE_PATH: Optional[str] = None

class SharedState:
    """Thread-safe shared state manager for recording operations."""
    
    def __init__(self) -> None:
        self.recording = False
        self.should_exit = False
        self.recording_count = 0
        self.current_output_dir: Optional[str] = None
        self.recording_start_time: Optional[float] = None
        self.recording_end_time: Optional[float] = None
        self.recording_duration: Optional[float] = None
        self.demo_directory_name: Optional[str] = None
        self.camera_stats: Dict[str, Dict] = {}
        self.pose_stats: Dict[str, Dict] = {}
        self.lock = threading.Lock()
    
    def _backup_config_files(self, task_dir: str) -> None:
        """Backup config files to task directory (saved once at task level only)"""
        try:
            config_backup_dir = os.path.join(task_dir, "config_backup")
            
            if os.path.exists(config_backup_dir):
                print(f"[CONFIG] Config file already exists in task directory, skipping backup: {os.path.basename(config_backup_dir)}")
                return
            
            os.makedirs(config_backup_dir, exist_ok=True)
            
            # Backup config files
            global CONFIG_FILE_PATH, CONFIG
            if CONFIG_FILE_PATH and os.path.exists(CONFIG_FILE_PATH):
                config_backup_path = os.path.join(config_backup_dir, "data_collection.yaml")
                shutil.copy2(CONFIG_FILE_PATH, config_backup_path)
                print(f"[CONFIG] Backed up config file: {os.path.basename(CONFIG_FILE_PATH)} -> {task_dir}/config_backup/")
            
            vitamin_config_path = os.path.join(ROOT_DIR, "config", "vitamin_b_config.py")
            if os.path.exists(vitamin_config_path):
                vitamin_backup_path = os.path.join(config_backup_dir, "vitamin_b_config.py")
                shutil.copy2(vitamin_config_path, vitamin_backup_path)
                print(f"[CONFIG] Backed up config file: vitamin_b_config.py -> {task_dir}/config_backup/")
            
            if CONFIG:
                runtime_config_path = os.path.join(config_backup_dir, "runtime_config.yaml")
                with open(runtime_config_path, 'w', encoding='utf-8') as f:
                    OmegaConf.save(CONFIG, f)
                print(f"[CONFIG] Saved runtime config snapshot: runtime_config.yaml -> {task_dir}/config_backup/")
            
            backup_info = {
                "backup_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "original_config_file": CONFIG_FILE_PATH,
                "vitamin_config_file": vitamin_config_path,
                "task_name": CONFIG.task.name if CONFIG else "unknown",
                "task_type": CONFIG.task.type if CONFIG else "unknown"
            }
            
            backup_info_path = os.path.join(config_backup_dir, "backup_info.json")
            with open(backup_info_path, 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, indent=4, ensure_ascii=False)
            
            print(f"[CONFIG] Config file backup complete: {config_backup_dir}")
            
        except Exception as e:
            print(f"[WARNING] Config file backup failed: {e}")
    
    def start_recording(self) -> Tuple[str, float]:
        """Start recording and create output directory."""
        with self.lock:
            self.recording = True
            self.recording_count += 1
            self.recording_start_time = time.time()
            
            timestamp = datetime.now().strftime(r'%Y.%m.%d_%H.%M.%S.%f')
            
            # Use global CONFIG
            global CONFIG
            base_dir = CONFIG.recorder.output
            task_name = CONFIG.task.name
            task_type = CONFIG.task.type
            
            task_dir = os.path.join(base_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            
            self._backup_config_files(task_dir)
            
            self.current_output_dir = os.path.join(task_dir, f"demo_{task_type}_{timestamp}")
            self.demo_directory_name = f"demo_{task_type}_{timestamp}"
            os.makedirs(self.current_output_dir, exist_ok=True)
            
            print("\n" + "="*60)
            print(f"🎬 STARTING DEMO #{self.recording_count:02d} 🎬")
            print("="*60)
            print(f"[RECORDING] Demo directory: {self.demo_directory_name}")
            print("="*60)
            
            return self.current_output_dir, self.recording_start_time
    
    def stop_recording(self) -> Optional[str]:
        """Stop recording and return output directory."""
        with self.lock:
            self.recording = False
            self.recording_end_time = time.time()
            if self.recording_start_time:
                self.recording_duration = self.recording_end_time - self.recording_start_time
            
            print("\n" + "="*60)
            print(f"🛑 DEMO #{self.recording_count:02d} COMPLETED 🛑")
            print("="*60)
            return self.current_output_dir
    
    def save_camera_stats(self, stats: Dict[str, Dict]) -> None:
        """Save camera statistics."""
        with self.lock:
            self.camera_stats = stats
    
    def save_pose_stats(self, stats: Dict[str, Dict]) -> None:
        """Save pose statistics."""
        with self.lock:
            self.pose_stats = stats
    
    def display_summary_and_save_details(self) -> None:
        """Display recording summary and save detailed log to file."""
        with self.lock:
            if not self.recording_start_time or not self.recording_end_time:
                return
            
            start_time_str = datetime.fromtimestamp(self.recording_start_time).strftime('%H:%M:%S')
            end_time_str = datetime.fromtimestamp(self.recording_end_time).strftime('%H:%M:%S')
            
            print(f"\n📊 Demo #{self.recording_count:02d} Recording Summary")
            print(f"\n🎬 Recording Information")
            print(f"- Recording Number: Demo #{self.recording_count:02d}")
            print(f"- Start Time: {start_time_str}")
            print(f"- End Time: {end_time_str}")
            print(f"- Recording Duration: {self.recording_duration:.2f} seconds")
            
            if self.camera_stats:
                active_cameras = []
                total_frames = 0
                avg_fps = 0
                for cam_name, stats in self.camera_stats.items():
                    usage_name = bimanual_usage_map.get("left_hand", {}).get(cam_name.split('_')[-1], cam_name) if "left_" in cam_name else bimanual_usage_map.get("right_hand", {}).get(cam_name.split('_')[-1], cam_name)
                    resolution = bimanual_cam_list.get("left_hand" if "left_" in cam_name else "right_hand", {}).get(cam_name.split('_')[-1], {}).get("res", "unknown")
                    width, height = self._get_camera_dimensions(resolution)
                    active_cameras.append(f"{usage_name} ({width}x{height})")
                    total_frames += stats.get('frames', 0)
                    avg_fps += stats.get('fps', 0)
                
                avg_fps = avg_fps / len(self.camera_stats) if self.camera_stats else 0
                
                print(f"\n📹 Camera Configuration")
                print(f"- Active Cameras: {', '.join(active_cameras)}")
                
                print(f"\n📈 Overall Statistics")
                print(f"- Total Frames: {total_frames//len(self.camera_stats) if self.camera_stats else 0} frames per camera")
                print(f"- Frame Rate: {avg_fps:.1f} fps (all cameras)")
            
            if self.pose_stats:
                if 'left' in self.pose_stats:
                    stats = self.pose_stats['left']
                    print(f"- Pose Data: {stats.get('frames', 0)} frames, {stats.get('fps', 0):.1f} fps, stability {stats.get('stability', 'Unknown')}")
                elif 'right' in self.pose_stats:
                    stats = self.pose_stats['right']
                    print(f"- Pose Data: {stats.get('frames', 0)} frames, {stats.get('fps', 0):.1f} fps, stability {stats.get('stability', 'Unknown')}")
            
            print(f"\n✅ Completion Status")
            print(f"All recording components completed successfully, data saved to `{self.demo_directory_name}` directory")
            
            self._save_detailed_log()
            print(f"\n📄 Detailed information saved to: `demo_{self.recording_count:02d}_detailed_log.txt`")
    
    def _get_camera_dimensions(self, resolution: str) -> Tuple[int, int]:
        """Get camera dimensions based on resolution string."""
        resolution_map = {
            "400": (640, 400),
            "480": (640, 480),
            "720": (1280, 720),
            "800": (1280, 800),
            "1080": (1920, 1080),
            "1200": (1920, 1200),
        }
        return resolution_map.get(resolution, (1280, 720))
    
    def _save_detailed_log(self) -> None:
        """Save detailed recording log to file."""
        if not self.current_output_dir:
            return
        
        log_filename = f"demo_{self.recording_count:02d}_detailed_log.txt"
        log_path = os.path.join(self.current_output_dir, log_filename)
        
        start_time_str = datetime.fromtimestamp(self.recording_start_time).strftime('%H:%M:%S')
        end_time_str = datetime.fromtimestamp(self.recording_end_time).strftime('%H:%M:%S')
        current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"🎬 DEMO #{self.recording_count:02d} DETAILED LOG 🎬\n")
            f.write("="*60 + "\n")
            f.write(f"Recording Directory: {self.demo_directory_name}\n")
            f.write(f"Recording Start Time: {start_time_str}\n")
            f.write(f"Recording End Time: {end_time_str}\n")
            f.write(f"Total Duration: {self.recording_duration:.2f}s\n\n")
            
            # Continue with detailed log content...
            f.write("="*60 + "\n")
            f.write("CAMERA CONFIGURATION\n")
            f.write("="*60 + "\n")
            f.write("Active Cameras:\n")
            for cam_name in self.camera_stats.keys():
                f.write(f"- {cam_name}\n")
            f.write("\n")
            
            f.write("="*60 + "\n")
            f.write("RECORDING STATISTICS\n")
            f.write("="*60 + "\n")
            for cam_name, stats in self.camera_stats.items():
                f.write(f"{cam_name}:\n")
                f.write(f"- Frames captured: {stats.get('frames', 0)}\n")
                f.write(f"- Frame rate: {stats.get('fps', 0):.1f} fps\n")
                f.write(f"- Timestamps saved: {stats.get('timestamps', 0)}\n\n")
            
            f.write("="*60 + "\n")
            f.write(f"Generated at: {current_time_str}\n")
            f.write("="*60 + "\n")

    def is_recording(self) -> bool:
        """Check if currently recording."""
        with self.lock:
            return self.recording

    def get_recording_count(self) -> int:
        """Get current recording count."""
        with self.lock:
            return self.recording_count
    
    def get_current_output_dir(self) -> Optional[str]:
        """Get current output directory."""
        with self.lock:
            return self.current_output_dir
    
    def set_exit(self) -> None:
        """Signal threads to exit."""
        with self.lock:
            self.should_exit = True
    
    def should_quit(self) -> bool:
        """Check if threads should exit."""
        with self.lock:
            return self.should_exit
    
    def get_recording_start_time(self) -> Optional[float]:
        """Get recording start time for timestamp synchronization."""
        with self.lock:
            return self.recording_start_time


class BimanualVideoRecorder:
    """Handles video recording operations for bimanual setup with multiple cameras."""
    
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.writers: Dict[str, cv2.VideoWriter] = {}
        self.recording = False
        self.start_time: Optional[float] = None
        self.frame_counts: Dict[str, int] = {}
        self.frame_timestamps: Dict[str, List[float]] = {}
        self.writer_locks: Dict[str, threading.Lock] = {}  # Independent lock for each camera
        self.state_lock = threading.Lock()  # Only protects state changes
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_camera_dimensions(self, resolution: str) -> Tuple[int, int]:
        """Get camera dimensions based on resolution string."""
        resolution_map = {
            "400": (640, 400),
            "480": (640, 480),
            "720": (1280, 720),
            "800": (1280, 800),
            "1080": (1920, 1080),
            "1200": (1920, 1200),
        }
        return resolution_map.get(resolution, (1280, 720))
    
    def start_recording(self) -> None:
        """Start recording for bimanual cameras."""
        with self.state_lock:  # Only protects state changes
            if self.recording:
                print("[WARNING] Already recording, ignoring start command")
                return
            
            self.start_time = time.time()
            timestamp_str = datetime.fromtimestamp(self.start_time).strftime('%H:%M:%S')
            
            print(f"[RECORDING] Bimanual video capture started at {timestamp_str}")
            
            active_cameras = []
            for hand_type, hand_cam_list in bimanual_cam_list.items():
                for cam_name, cam_props in hand_cam_list.items():
                    width, height = self._get_camera_dimensions(cam_props["res"])
                    
                    usage_name = bimanual_usage_map[hand_type][cam_name]
                    camera_id = f"{hand_type}_{cam_name}"
                    filename = f"{self.output_dir}/{usage_name}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*CODEC)
                    is_color = cam_props["color"]
                    
                    self.writers[camera_id] = cv2.VideoWriter(
                        filename, fourcc, FPS, (width, height), isColor=is_color
                    )
                    
                    # Create independent lock for each camera
                    self.writer_locks[camera_id] = threading.Lock()
                    self.frame_counts[camera_id] = 0
                    self.frame_timestamps[camera_id] = []
                    
                    active_cameras.append(f"{usage_name} ({width}x{height})")
            
            print(f"[RECORDING] Active cameras: {', '.join(active_cameras)}")
            self.recording = True
    
    def stop_recording(self) -> Dict[str, Dict]:
        """Stop recording all bimanual cameras and return statistics."""
        with self.state_lock:  # Only protects state changes
            if not self.recording:
                print("[WARNING] Not currently recording, ignoring stop command")
                return {}
            
            # Set recording to False first to prevent new writes
            self.recording = False
            
            end_time = time.time()
            duration = end_time - self.start_time if self.start_time else 0
            timestamp_str = datetime.fromtimestamp(end_time).strftime('%H:%M:%S')
            
            print(f"[RECORDING] Bimanual video capture stopped at {timestamp_str} (duration: {duration:.2f}s)")
        
        # Safely close writers outside state lock to avoid blocking other operations
        camera_stats = {}
        
        # Sequentially acquire lock for each camera and close writer
        for camera_id in list(self.writers.keys()):
            writer_lock = self.writer_locks.get(camera_id)
            if writer_lock:
                with writer_lock:
                    writer = self.writers.get(camera_id)
                    if writer:
                        try:
                            writer.release()
                        except Exception as e:
                            print(f"[WARNING] Error releasing writer for {camera_id}: {e}")
                    
                    frames = self.frame_counts.get(camera_id, 0)
                    
                    self._save_timestamps(camera_id)
                    
                    actual_fps = frames / duration if duration > 0 else 0
                    
                    parts = camera_id.split('_')
                    if len(parts) >= 3:
                        hand_type = f"{parts[0]}_hand"
                        cam_name = '_'.join(parts[2:])
                    else:
                        hand_type, cam_name = camera_id.split('_', 1)
                    usage_name = bimanual_usage_map[hand_type][cam_name]
                    
                    camera_stats[camera_id] = {
                        'frames': frames,
                        'fps': actual_fps,
                        'timestamps': len(self.frame_timestamps.get(camera_id, [])),
                        'usage_name': usage_name
                    }
                    
                    print(f"[RECORDING] {usage_name}: {frames} frames, {actual_fps:.1f} fps")
        
        # Final cleanup of state
        with self.state_lock:
            self.writers = {}
            self.writer_locks = {}
            self.frame_counts = {}
            self.frame_timestamps = {}
            print("[RECORDING] Bimanual video recording completed successfully")
            
        return camera_stats
    
    def _save_timestamps(self, camera_id: str) -> None:
        """Save timestamps for a specific camera to CSV file."""
        timestamps = self.frame_timestamps.get(camera_id, [])
        if not timestamps:
            return
        
        timestamp_df = pd.DataFrame({
            "frame_number": range(1, len(timestamps) + 1),
            "timestamp": timestamps
        })
        
        parts = camera_id.split('_')
        if len(parts) >= 3:
            hand_type = f"{parts[0]}_hand"
            cam_name = '_'.join(parts[2:])
        else:
            hand_type, cam_name = camera_id.split('_', 1)
        usage_name = bimanual_usage_map[hand_type][cam_name]
        
        timestamp_file = f"{self.output_dir}/{usage_name}_timestamps.csv"
        timestamp_df.to_csv(timestamp_file, index=False)
        print(f"[DATA] Saved {len(timestamps)} timestamps for {usage_name}")
    
    def write_frame(self, camera_id: str, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Write a frame to the specified camera's video file."""
        # Quick check to avoid unnecessary lock contention
        if not self.recording or camera_id not in self.writers:
            return
        
        # Only lock for current camera, allow other cameras to write in parallel
        writer_lock = self.writer_locks.get(camera_id)
        if writer_lock is None:
            return
            
        with writer_lock:
            # Check state again (double-check pattern)
            if not self.recording or camera_id not in self.writers:
                return
            
            self.writers[camera_id].write(frame)
            self.frame_counts[camera_id] += 1
            
            if timestamp is not None:
                self.frame_timestamps[camera_id].append(timestamp)
            else:
                self.frame_timestamps[camera_id].append(time.time())
    
    def get_recording_status(self, camera_id: Optional[str] = None) -> Tuple[str, float]:
        """Get recording status information for display."""
        if not self.recording:
            return "", 0
        
        current_time = time.time()
        duration = current_time - self.start_time if self.start_time else 0
        current_time_str = datetime.fromtimestamp(current_time).strftime('%H:%M:%S')
        
        if camera_id and camera_id in self.frame_counts:
            frames = self.frame_counts[camera_id]
            return f"REC {current_time_str} ({frames} frames)", duration
        
        return f"REC {current_time_str}", duration


class PoseDataCollector:
    """Collects pose data from Quest VR controllers."""
    
    def __init__(self, shared_state: SharedState, task_type: str = "bimanual") -> None:
        self.shared_state = shared_state
        self.task_type = task_type
        self.quest = None
        self.reset_trajectories()
    
    def reset_trajectories(self) -> None:
        """Reset trajectory data storage."""
        self.current_trajectory_left: List[List[float]] = []
        self.current_trajectory_right: List[List[float]] = []
        self.trajectory_start_time: Optional[float] = None
    
    def start_collection(self) -> None:
        """Initialize Quest module connection."""
        self.quest = QuestBimanualModule(
            VR_HOST, LOCAL_HOST, POSE_CMD_PORT, IK_RESULT_PORT, vis_sp=None
        )
        print(f"[QUEST] Initialized bimanual pose tracking for {self.task_type} arm mode")
    
    def _collect_pose_data(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Collect pose data from Quest module."""
        try:
            wrist_left, wrist_right, _ = self.quest.receive()
            
            if wrist_left is not None or wrist_right is not None:
                if not hasattr(self, '_quest_connected') or not self._quest_connected:
                    print("[QUEST] Connection established with Quest device")
                    self._quest_connected = True
            else:
                if hasattr(self, '_quest_connected') and self._quest_connected:
                    if not hasattr(self, '_last_data_time'):
                        self._last_data_time = time.time()
                    elif time.time() - self._last_data_time > 5.0:
                        print("[QUEST] No data received for 5 seconds - Quest may be disconnected")
                        self._quest_connected = False
                
            if wrist_left is not None or wrist_right is not None:
                self._last_data_time = time.time()
            
            return wrist_left, wrist_right
            
        except Exception as e:
            print(f"[QUEST] Error collecting pose data: {e}")
            return None, None
    
    def _store_pose_data(self, wrist_left: Optional[Any], wrist_right: Optional[Any], pose_timestamp: float) -> None:
        """Store pose data with timestamp."""
        if wrist_left is not None:
            pos, rot = wrist_left[0], wrist_left[1]
            self.current_trajectory_left.append([
                pose_timestamp, pos[0], pos[1], pos[2],
                rot[0], rot[1], rot[2], rot[3]
            ])
        
        if wrist_right is not None:
            pos, rot = wrist_right[0], wrist_right[1]
            self.current_trajectory_right.append([
                pose_timestamp, pos[0], pos[1], pos[2],
                rot[0], rot[1], rot[2], rot[3]
            ])
    
    def collect_data_thread(self) -> None:
        """Thread function to continuously collect pose data."""
        try:
            print("[QUEST] Starting pose data collection thread")
            self.start_collection()
            self.reset_trajectories()
            
            while not self.shared_state.should_quit():
                pose_timestamp = time.time()
                
                wrist_left, wrist_right = self._collect_pose_data()
                
                if self.shared_state.is_recording():
                    if self.trajectory_start_time is None:
                        self.trajectory_start_time = pose_timestamp
                    
                    self._store_pose_data(wrist_left, wrist_right, pose_timestamp)
                else:
                    if self.trajectory_start_time is not None:
                        self.save_trajectory_data()
                        self.reset_trajectories()
                
                time.sleep(0.01)
        
        except Exception as e:
            print(f"[ERROR] Pose data collection failed: {e}")
        
        finally:
            if (self.trajectory_start_time is not None and 
                (self.current_trajectory_left or self.current_trajectory_right)):
                self.save_trajectory_data()
            print("[QUEST] Pose data collection thread terminated")
    
    def save_trajectory_data(self) -> None:
        """Save collected trajectory data to files."""
        output_dir = self.shared_state.get_current_output_dir()
        if not output_dir:
            print("[WARNING] No output directory available, skipping pose data save")
            return
        
        pose_dir = os.path.join(output_dir, "pose_data")
        os.makedirs(pose_dir, exist_ok=True)
        
        demo_num = self.shared_state.get_recording_count()
        print(f"[QUEST] Saving pose data for Demo #{demo_num:02d}: {pose_dir}")
        
        pose_stats = {}
        
        if self.current_trajectory_left:
            left_stats = self._save_hand_trajectory(self.current_trajectory_left, "left", pose_dir)
            if left_stats:
                pose_stats['left'] = left_stats
        
        if self.current_trajectory_right:
            right_stats = self._save_hand_trajectory(self.current_trajectory_right, "right", pose_dir)
            if right_stats:
                pose_stats['right'] = right_stats
        
        self.shared_state.save_pose_stats(pose_stats)
    
    def _save_hand_trajectory(self, trajectory_data: List[List[float]], 
                            hand: str, pose_dir: str) -> Dict[str, float]:
        """Save trajectory data for a specific hand."""
        if not trajectory_data:
            return {}
        
        trajectory_df = pd.DataFrame(
            trajectory_data,
            columns=["timestamp", "x", "y", "z", "q_x", "q_y", "q_z", "q_w"]
        )
        trajectory_file = os.path.join(pose_dir, f"{hand}_hand_trajectory.csv")
        trajectory_df.to_csv(trajectory_file, index=False)
        
        total_duration = trajectory_data[-1][0] - trajectory_data[0][0]
        total_frames = len(trajectory_data)
        average_fps = total_frames / total_duration if total_duration > 0 else 0
        
        metadata = {
            "position": hand,
            "start_time": trajectory_data[0][0],
            "end_time": trajectory_data[-1][0],
            "duration": total_duration,
            "num_frames": total_frames,
            "average_fps": round(average_fps, 2),
            "datetime": datetime.now().strftime(r'%Y.%m.%d_%H:%M:%S.%f')
        }
        
        metadata_file = os.path.join(pose_dir, f"{hand}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"[QUEST] {hand.capitalize()} hand: {total_frames} frames, "
              f"{total_duration:.2f}s, {average_fps:.1f} fps")
        
        return {
            'frames': total_frames,
            'duration': total_duration,
            'fps': average_fps,
        }


def load_config(config_path: str, overrides: List[str]) -> OmegaConf:
    """Load and merge configuration from file and command line overrides."""
    try:
        config = OmegaConf.load(config_path)
        # Apply any command line overrides
        if overrides:
            override_config = OmegaConf.from_dotlist(overrides)
            config = OmegaConf.merge(config, override_config)
        return config
    except Exception as e:
        print(f"[ERROR] Failed to load config file '{config_path}': {e}")
        raise


def print_instructions(title: str = "ViTaMIn-B Data Recorder v1.0") -> None:
    """Print user instructions."""
    print("\n" + "="*60)
    print(f"       {title}")
    print("="*60)
    print("CONTROLS:")
    print("  [Q]     - Quit application")
    print("  [S/s]   - Toggle recording (start/stop)")
    print("="*60)
    print("DISPLAY:")
    print("  • Video Windows: Demo number shown in top-left corner")
    print("  • Console: Demo start/stop messages with bright banners")
    print("  • FPS/Quality: Real-time performance monitoring")
    print("="*60)
    print("STATUS: Ready to record")
    print()


def get_camera_dimensions(resolution: str) -> Tuple[int, int]:
    """Get camera dimensions based on resolution string."""
    resolution_map = {
        "400": (640, 400),
        "480": (640, 480),
        "720": (1280, 720),
        "800": (1280, 800),
        "1080": (1920, 1080),
        "1200": (1920, 1200),
    }
    return resolution_map.get(resolution, (1280, 800))  # Default to 720p 