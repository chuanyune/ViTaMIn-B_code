# coding=utf-8

"""
THREAD SAFETY NOTICE:
This module has been specifically designed to avoid OpenCV Segmentation Faults
by following strict thread safety rules:

1. [SUCCESS] Each USBCamWorker thread owns ONLY ONE VideoCapture object
2. [SUCCESS] VideoWriter objects are managed ONLY in the main thread via VideoRecorder
3. [SUCCESS] Worker threads only capture frames and put them in queues
4. [SUCCESS] Main thread reads from queues and writes to VideoWriter objects
5. [SUCCESS] No sharing of OpenCV objects between threads

[ERROR] DO NOT:
- Share VideoCapture between threads
- Share VideoWriter between threads  
- Access the same OpenCV object from multiple threads simultaneously

[SUCCESS] SAFE PATTERN:
Thread A: VideoCapture.read() -> Queue
Thread B: Queue.get() -> VideoWriter.write()
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import select
import tty
import termios

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from queue import Queue

# Add the root directory to the path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
print(f"[SYSTEM] Root directory: {ROOT_DIR}")

from utils.fps_utils import FPSHandler

# Global configuration
CONFIG: Optional[OmegaConf] = None
CONFIG_FILE_PATH: Optional[str] = None  # Store the config file path for backup

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
        self.lock = threading.Lock()
        self.last_summary_demo_count: int = 0
        
        # Demo statistics
        self.demo_stats: List[Dict[str, Any]] = []  # Store statistics for each demo

    def _backup_config_files(self, task_dir: str) -> None:
        """Backup configuration files to the task directory (only once per task)."""
        try:
            config_backup_dir = os.path.join(task_dir, "config_backup")
            all_traj_dir = os.path.join(task_dir, "all_trajectory")

            if os.path.exists(config_backup_dir):
                print(f"[CONFIG] Config files already backed up, skipping: {os.path.basename(config_backup_dir)}")
                return

            os.makedirs(config_backup_dir, exist_ok=True)
            os.makedirs(all_traj_dir, exist_ok=True)

            if CONFIG_FILE_PATH and os.path.exists(CONFIG_FILE_PATH):
                config_backup_path = os.path.join(config_backup_dir, os.path.basename(CONFIG_FILE_PATH))
                shutil.copy2(CONFIG_FILE_PATH, config_backup_path)
                print(f"[CONFIG] Backed up config file: {os.path.basename(CONFIG_FILE_PATH)} -> {task_dir}/config_backup/")

            if CONFIG is not None:
                runtime_config_path = os.path.join(config_backup_dir, "runtime_config.yaml")
                with open(runtime_config_path, 'w', encoding='utf-8') as f:
                    OmegaConf.save(CONFIG, f)
                print(f"[CONFIG] Saved runtime config snapshot: runtime_config.yaml -> {task_dir}/config_backup/")

            backup_info = {
                "backup_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "original_config_file": CONFIG_FILE_PATH,
                "task_name": CONFIG.task.name if CONFIG else "unknown",
                "task_type": CONFIG.task.type if CONFIG else "unknown"
            }

            backup_info_path = os.path.join(config_backup_dir, "backup_info.json")
            with open(backup_info_path, 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, indent=4, ensure_ascii=False)

            print(f"[CONFIG] Config file backup completed: {config_backup_dir}")

        except Exception as e:
            print(f"[WARNING] Failed to backup config files: {e}")



    def start_recording(self) -> Tuple[str, float]:
        """Start recording and create output directory."""
        with self.lock:
            self.recording = True
            self.recording_count += 1
            self.recording_start_time = time.time()

            timestamp = datetime.now().strftime(r'%Y.%m.%d_%H.%M.%S.%f')

            if not hasattr(CONFIG, 'recorder') or not hasattr(CONFIG.recorder, 'output'):
                raise ValueError("[CONFIG] Missing required 'recorder.output'")
            if not hasattr(CONFIG, 'task') or not hasattr(CONFIG.task, 'name') or not hasattr(CONFIG.task, 'type'):
                raise ValueError("[CONFIG] Missing required 'task.name' or 'task.type'")
            base_dir = CONFIG.recorder.output
            task_name = CONFIG.task.name
            task_type = CONFIG.task.type

            task_dir = os.path.join(base_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)



            self._backup_config_files(task_dir)

            self.current_output_dir = os.path.join(task_dir, "demos", f"demo_{task_type}_{timestamp}")
            self.demo_directory_name = f"demo_{task_type}_{timestamp}"
            os.makedirs(self.current_output_dir, exist_ok=True)

            print("\n" + "="*60)
            print(f"STARTING DEMO #{self.recording_count:02d}")
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
            print(f"DEMO #{self.recording_count:02d} COMPLETED")
            print("="*60)
            return self.current_output_dir

    def save_camera_stats(self, stats: Dict[str, Dict]) -> None:
        """Save camera statistics."""
        with self.lock:
            for cam_name, cam_stats in stats.items():
                required_keys = ['frames', 'fps', 'timestamps', 'width', 'height']
                missing_keys = [key for key in required_keys if key not in cam_stats]
                if missing_keys:
                    raise ValueError(f"[ERROR] Missing required stats {missing_keys} for camera {cam_name}")
            self.camera_stats = stats

    def display_summary_and_save_details(self) -> None:
        """Display recording summary and save detailed log to file."""
        with self.lock:
            if not self.recording_start_time or not self.recording_end_time:
                return
            if self.recording_count == self.last_summary_demo_count:
                return

            start_time_str = datetime.fromtimestamp(self.recording_start_time).strftime('%H:%M:%S')
            end_time_str = datetime.fromtimestamp(self.recording_end_time).strftime('%H:%M:%S')

            print(f"\n Demo #{self.recording_count:02d} Summary")
            print(f"\n Recording Information")
            print(f"- Recording ID: Demo #{self.recording_count:02d}")
            print(f"- Start Time: {start_time_str}")
            print(f"- End Time: {end_time_str}")
            print(f"- Duration: {self.recording_duration:.2f}s")

            if self.camera_stats:
                active_cameras = []
                total_frames = 0
                for cam_name, stats in self.camera_stats.items():
                    if 'width' not in stats or 'height' not in stats:
                        raise ValueError(f"[ERROR] Missing 'width' or 'height' in camera stats for {cam_name}")
                    width, height = stats['width'], stats['height']
                    active_cameras.append(f"{cam_name} ({width}x{height})")
                    total_frames += stats.get('frames', 0)
                    
                print(f"\n Camera Configuration")
                print(f"- Active Cameras: {', '.join(active_cameras)}")
                print(f"\n Camera Statistics")
                for cam_name, stats in self.camera_stats.items():
                    print(f"- {cam_name}: {stats['frames']} frames, {stats['fps']:.1f} fps")

            print(f"\n[SUCCESS] Completion Status")
            print(f"All recording components completed successfully, data saved to `{self.demo_directory_name}`")

            # Save current demo statistics
            demo_info = {
                'demo_number': self.recording_count,
                'demo_name': self.demo_directory_name,
                'start_time': self.recording_start_time,
                'end_time': self.recording_end_time,
                'duration': self.recording_duration,
                'start_time_str': start_time_str,
                'end_time_str': end_time_str,
                'total_frames': total_frames,
                'camera_count': len(self.camera_stats),
                'camera_stats': self.camera_stats.copy()
            }
            self.demo_stats.append(demo_info)

            self._save_detailed_log()
            print(f"\n Detailed log saved to: `demo_{self.recording_count:02d}_detailed_log.txt`")
            self.last_summary_demo_count = self.recording_count

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
            f.write(f"DEMO #{self.recording_count:02d} DETAILED LOG\n")
            f.write("="*60 + "\n")
            f.write(f"Recording Directory: {self.demo_directory_name}\n")
            f.write(f"Recording Start Time: {start_time_str}\n")
            f.write(f"Recording End Time: {end_time_str}\n")
            f.write(f"Total Duration: {self.recording_duration:.2f}s\n\n")

            f.write("="*60 + "\n")
            f.write("CAMERA CONFIGURATION\n")
            f.write("="*60 + "\n")
            f.write("Active Cameras:\n")
            for cam_name, stats in self.camera_stats.items():
                if 'width' not in stats or 'height' not in stats:
                    raise ValueError(f"[ERROR] Missing 'width' or 'height' in camera stats for {cam_name}")
                width, height = stats['width'], stats['height']
                f.write(f"- {cam_name} ({width}x{height})\n")
            f.write("\n")

            f.write("="*60 + "\n")
            f.write("RECORDING STATISTICS\n")
            f.write("="*60 + "\n")
            for cam_name, stats in self.camera_stats.items():
                f.write(f"{cam_name} Camera:\n")
                f.write(f"- Frames captured: {stats['frames']}\n")
                f.write(f"- Frame rate: {stats['fps']:.1f} fps\n")
                f.write(f"- Timestamps saved: {stats['timestamps']}\n\n")

            f.write("="*60 + "\n")
            f.write("SESSION COMPLETION STATUS\n")
            f.write("="*60 + "\n")
            f.write("[SUCCESS] Video recording completed successfully\n")
            f.write("[SUCCESS] Camera windows closed\n")
            f.write("[SUCCESS] Data recording session completed successfully\n")
            f.write("[SUCCESS] All threads finished successfully\n\n")

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
    
    def generate_final_demo_statistics(self) -> None:
        """Generate and save final demo statistics report"""
        with self.lock:
            if not self.demo_stats:
                print("\n[STATS] No demo statistics data")
                return
            
            # Calculate statistics
            total_demos = len(self.demo_stats)
            durations = [demo['duration'] for demo in self.demo_stats if demo['duration']]
            
            if not durations:
                print("\n[STATS] No valid duration data")
                return
            
            # Find longest and shortest demos
            longest_demo = max(self.demo_stats, key=lambda x: x['duration'])
            shortest_demo = min(self.demo_stats, key=lambda x: x['duration'])
            
            # Calculate average duration
            avg_duration = sum(durations) / len(durations)
            total_duration = sum(durations)
            
            # Generate statistics report
            stats_report = {
                "session_summary": {
                    "total_demos": total_demos,
                    "total_duration": round(total_duration, 2),
                    "average_duration": round(avg_duration, 2),
                    "longest_demo": {
                        "demo_number": longest_demo['demo_number'],
                        "demo_name": longest_demo['demo_name'],
                        "duration": round(longest_demo['duration'], 2)
                    },
                    "shortest_demo": {
                        "demo_number": shortest_demo['demo_number'],
                        "demo_name": shortest_demo['demo_name'],
                        "duration": round(shortest_demo['duration'], 2)
                    }
                },
                "demo_details": self.demo_stats,
                "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Print statistics results
            print("\n" + "="*80)
            print(" Final Demo Statistics Report")
            print("="*80)
            print(f"Total Demos: {total_demos}")
            print(f"Total Recording Duration: {total_duration:.2f}s ({total_duration/60:.1f} minutes)")
            print(f"Average Duration: {avg_duration:.2f}s")
            print(f"Longest Demo: #{longest_demo['demo_number']:02d} ({longest_demo['duration']:.2f}s) - {longest_demo['demo_name']}")
            print(f"Shortest Demo: #{shortest_demo['demo_number']:02d} ({shortest_demo['duration']:.2f}s) - {shortest_demo['demo_name']}")
            print("="*80)
            
            # Save statistics file
            try:
                # Save to task root directory
                if hasattr(CONFIG, 'recorder') and hasattr(CONFIG.recorder, 'output') and hasattr(CONFIG, 'task'):
                    task_dir = os.path.join(CONFIG.recorder.output, CONFIG.task.name)
                    stats_file = os.path.join(task_dir, "session_statistics.json")
                    
                    with open(stats_file, 'w', encoding='utf-8') as f:
                        json.dump(stats_report, f, indent=4, ensure_ascii=False)
                    
                    print(f"Statistics report saved: {stats_file}")
                    
                    # Also save simplified text report
                    text_file = os.path.join(task_dir, "session_summary.txt")
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write("="*80 + "\n")
                        f.write(" Demo Recording Session Statistics Report\n")
                        f.write("="*80 + "\n")
                        f.write(f"Task Name: {CONFIG.task.name}\n")
                        f.write(f"Task Type: {CONFIG.task.type}\n")
                        f.write(f"Generated At: {stats_report['generated_at']}\n\n")
                        
                        f.write("Core Statistics:\n")
                        f.write(f"  - Total Demos: {total_demos}\n")
                        f.write(f"  - Total Recording Duration: {total_duration:.2f}s ({total_duration/60:.1f} minutes)\n")
                        f.write(f"  - Average Duration: {avg_duration:.2f}s\n\n")
                        
                        f.write("Extreme Values:\n")
                        f.write(f"  - Longest Demo: #{longest_demo['demo_number']:02d} ({longest_demo['duration']:.2f}s)\n")
                        f.write(f"    Directory: {longest_demo['demo_name']}\n")
                        f.write(f"  - Shortest Demo: #{shortest_demo['demo_number']:02d} ({shortest_demo['duration']:.2f}s)\n")
                        f.write(f"    Directory: {shortest_demo['demo_name']}\n\n")
                        
                        f.write("Demo Details:\n")
                        for demo in self.demo_stats:
                            f.write(f"  Demo #{demo['demo_number']:02d}: {demo['duration']:.2f}s ({demo['start_time_str']}-{demo['end_time_str']}) - {demo['demo_name']}\n")
                        
                        f.write("\n" + "="*80 + "\n")
                    
                    print(f"Text report saved: {text_file}")
                    
            except Exception as e:
                print(f"[ERROR] Failed to save statistics report: {e}")
            
            print(f"\nRecording session completed! Recorded {total_demos} demos, total duration {total_duration/60:.1f} minutes")


class USBCamWorker(threading.Thread):
    """Capture frames from a USB camera in a separate thread with dedicated queue."""

    def __init__(
        self,
        cam_name: str,
        device: Any,
        resolution: Tuple[int, int],
        fps: int,
        stop_event: threading.Event,
        queue_size: int = 32,
        headless: bool = False,
        auto_exposure: Optional[int] = None,
        exposure: Optional[int] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.device = device
        self.resolution = resolution
        self.fps = fps
        self.stop_event = stop_event
        self.headless = True  # Force headless mode
        
        # Exposure control parameters
        self.auto_exposure = auto_exposure
        self.exposure = exposure
        
        # Independent queue for each camera - key performance optimization
        self.frame_queue: Queue = Queue(maxsize=queue_size)
        self.cap: Optional[cv2.VideoCapture] = None
        self.actual_fps_handler = FPSHandler()
        
        # Remove direct recording - avoid multi-thread VideoWriter sharing
        # Only responsible for capture, not writing, writing handled by main thread
        
        # Performance statistics
        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_stats_time = time.time()

    def run(self) -> None:
        """Thread-safe: Only responsible for frame capture, not direct video writing"""
        try:
            # Use default backend, completely consistent with sadasd.py
            self.cap = cv2.VideoCapture(self.device)
            
            # Completely mimic sadasd.py camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            # Do not set FPS, let camera use default value (consistent with sadasd.py)
            
            # Tactile camera exposure control (only for tactile cameras)
            if 'tactile' in self.cam_name.lower() and self.auto_exposure is not None and self.exposure is not None:
                print(f"[EXPOSURE] Configuring exposure settings for tactile camera: {self.cam_name}")
                
                # Set auto exposure mode
                auto_exposure_result = self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.auto_exposure)
                if auto_exposure_result:
                    print(f"[EXPOSURE] {self.cam_name}: Auto exposure set to {self.auto_exposure}")
                else:
                    print(f"[WARNING] {self.cam_name}: Failed to set auto exposure")
                
                # Set exposure value
                exposure_result = self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
                if exposure_result:
                    actual_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
                    print(f"[EXPOSURE] {self.cam_name}: Exposure set to {self.exposure}, actual: {actual_exposure}")
                else:
                    print(f"[WARNING] {self.cam_name}: Failed to set exposure value")
                    
                print(f"[EXPOSURE] {self.cam_name}: Tactile camera exposure configuration completed")
            elif 'tactile' in self.cam_name.lower():
                print(f"[EXPOSURE] {self.cam_name}: Using default exposure settings (no config provided)")
            else:
                print(f"[EXPOSURE] {self.cam_name}: Visual camera - using automatic exposure")
            
            if not self.cap.isOpened():
                print(f"[ERROR] Failed to open camera {self.device} ({self.cam_name})")
                return
            
            # Verify camera settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Display exposure settings info (only for tactile cameras)
            if 'tactile' in self.cam_name.lower() and self.auto_exposure is not None and self.exposure is not None:
                actual_auto_exposure = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                actual_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
                print(f"[USB] {self.cam_name}: {actual_width}x{actual_height} @ {actual_fps}fps")
                print(f"[EXPOSURE] {self.cam_name}: Auto exposure: {actual_auto_exposure}, Exposure: {actual_exposure}")
            else:
                print(f"[USB] {self.cam_name}: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Simplified frame capture main loop - only capture, no writing
            fps_update_interval = 30
            last_fps_time = time.time()
            
            while not self.stop_event.is_set() and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print(f"[WARNING] {self.cam_name} Unable to get frame")
                    break
                
                self.frames_captured += 1
                timestamp = time.time()
                
                # Thread-safe: only put in queue, no direct video writing
                # Writing handled uniformly by main thread VideoRecorder
                try:
                    self.frame_queue.put((frame, timestamp), block=False)
                except:
                    self.frames_dropped += 1
                
                # FPS statistics (silent mode, no printing)
                if self.frames_captured % fps_update_interval == 0:
                    current_time = time.time()
                    elapsed_time = current_time - last_fps_time
                    current_fps = fps_update_interval / elapsed_time
                    # print(f"[{self.cam_name}] Real-time FPS: {current_fps:.2f} | Recorded: {self.frames_captured} frames")
                    last_fps_time = current_time
                
                # Key: add cv2.waitKey(1), completely consistent with sadasd.py
                # This is crucial for performance! But skip GUI calls in headless mode
                if not self.headless:
                    cv2.waitKey(1)
                    
        finally:
            # Thread-safe: only release VideoCapture, do not handle VideoWriter
            if self.cap:
                self.cap.release()
                print(f"[USB] {self.cam_name} released")
    
    def _print_stats_if_needed(self) -> None:
        """Periodically print performance statistics to avoid frequent output affecting performance"""
        current_time = time.time()
        if current_time - self.last_stats_time >= 5.0:  # Output every 5 seconds
            if self.frames_dropped > 0:
                drop_rate = (self.frames_dropped / self.frames_captured) * 100
                print(f"[PERF] {self.cam_name}: captured={self.frames_captured}, dropped={self.frames_dropped} ({drop_rate:.1f}%)")
            self.last_stats_time = current_time
    
    def get_frame_non_blocking(self) -> Optional[Tuple[np.ndarray, float]]:
        """Non-blocking get frame data"""
        try:
            return self.frame_queue.get(block=False)
        except:
            return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics"""
        return {
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'queue_size': self.frame_queue.qsize()
        }
    



class VideoRecorder:
    """Thread-safe: Handles video recording operations for multiple cameras in main thread only."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.writers: Dict[str, cv2.VideoWriter] = {}
        self.recording = False
        self.start_time: Optional[float] = None
        self.frame_counts: Dict[str, int] = {}
        self.frame_timestamps: Dict[str, List[float]] = {}
        self._last_width_height: Dict[str, Tuple[int, int]] = {}
        
        # Thread-safe: add lock to protect VideoWriter operations
        import threading
        self._lock = threading.Lock()

        os.makedirs(output_dir, exist_ok=True)
    
    def _create_unified_metadata(self, start_time: float, duration: float, camera_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """Create unified metadata dictionary for all cameras (similar to wireless version)."""
        # Calculate overall statistics
        total_frames = sum(stats['frames'] for stats in camera_stats.values())
        avg_fps = sum(stats['fps'] for stats in camera_stats.values()) / len(camera_stats) if camera_stats else 0
        
        # Determine position based on task type and cameras
        if not hasattr(CONFIG, 'task'):
            position = "unknown"
        else:
            task_type = getattr(CONFIG.task, 'type', 'single')
            if task_type == 'single':
                position = "left"  # Single mode uses left hand
            else:
                # For bimanual, determine if this is left or right based on camera names
                cam_names = list(camera_stats.keys())
                if any('left' in name.lower() for name in cam_names):
                    if any('right' in name.lower() for name in cam_names):
                        position = "bimanual"  # Both hands present
                    else:
                        position = "left"  # Only left hand cameras
                else:
                    position = "right"  # Only right hand cameras or unclear
        
        return {
            "recording_start_time": start_time,
            "position": position,
            "duration": duration,
            "num_frames": total_frames,
            "average_fps": round(avg_fps, 2),
            "datetime": datetime.fromtimestamp(start_time).strftime(r'%Y.%m.%d_%H:%M:%S.%f'),
            "cameras": camera_stats  # Include detailed camera stats for reference
        }
    
    def _save_unified_metadata(self, start_time: float, duration: float, camera_stats: Dict[str, Dict]) -> None:
        """Save unified metadata for all cameras."""
        metadata = self._create_unified_metadata(start_time, duration, camera_stats)
        metadata_filename = os.path.join(self.output_dir, "metadata.json")
        
        try:
            with open(metadata_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            print(f"[DATA] Saved unified metadata: {metadata_filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save unified metadata: {e}")

    def start_recording(self, cam_cfg: Dict[str, Dict[str, Any]]) -> None:
        """Start recording for specified cameras."""
        if self.recording:
            print("[WARNING] Already recording, ignoring start command")
            return

        for cam_name, props in cam_cfg.items():
            required_keys = ['width', 'height', 'color']
            missing_keys = [key for key in required_keys if key not in props]
            if missing_keys:
                raise ValueError(f"[CONFIG] Missing required camera properties {missing_keys} for {cam_name}")
            if not isinstance(props['width'], (int, float)) or not isinstance(props['height'], (int, float)):
                raise ValueError(f"[CONFIG] Invalid width or height for {cam_name}: must be numeric")
            if not isinstance(props['color'], bool):
                raise ValueError(f"[CONFIG] Invalid color for {cam_name}: must be boolean")

        self.start_time = time.time()
        timestamp_str = datetime.fromtimestamp(self.start_time).strftime('%H:%M:%S')

        print(f"[RECORDING] Video capture started at {timestamp_str}")

        active_cameras = []
        if not hasattr(CONFIG, 'recorder') or not hasattr(CONFIG.recorder, 'fps') or not hasattr(CONFIG.recorder, 'codec'):
            raise ValueError("[CONFIG] Missing required 'recorder.fps' or 'recorder.codec'")
        fps = int(CONFIG.recorder.fps)
        codec = str(CONFIG.recorder.codec)
        fourcc = cv2.VideoWriter_fourcc(*codec)

        for cam_name, props in cam_cfg.items():
            width = int(props['width'])
            height = int(props['height'])
            is_color = bool(props['color'])

            filename = f"{self.output_dir}/{cam_name}.mp4"
            self.writers[cam_name] = cv2.VideoWriter(
                filename, fourcc, fps, (width, height), isColor=is_color
            )
            self._last_width_height[cam_name] = (width, height)

            self.frame_counts[cam_name] = 0
            self.frame_timestamps[cam_name] = []

            active_cameras.append(f"{cam_name} ({width}x{height})")

        print(f"[RECORDING] Active cameras: {', '.join(active_cameras)}")
        self.recording = True

    def stop_recording(self) -> Dict[str, Dict[str, Any]]:
        """Stop recording all cameras and print statistics."""
        if not self.recording:
            print("[WARNING] Not currently recording, ignoring stop command")
            return {}

        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        timestamp_str = datetime.fromtimestamp(end_time).strftime('%H:%M:%S')

        print(f"[RECORDING] Video capture stopped at {timestamp_str} (duration: {duration:.2f}s)")

        camera_stats: Dict[str, Dict[str, Any]] = {}

        for cam_name, writer in self.writers.items():
            writer.release()
            frames = self.frame_counts[cam_name]

            self._save_timestamps(cam_name)

            actual_fps = frames / duration if duration > 0 else 0
            print(f"[RECORDING] {cam_name}: {frames} frames, {actual_fps:.1f} fps")

            if cam_name not in self._last_width_height:
                raise ValueError(f"[ERROR] Missing width/height for {cam_name} in stop_recording")
            width, height = self._last_width_height[cam_name]
            camera_stats[cam_name] = {
                'frames': frames,
                'fps': actual_fps,
                'average_fps': round(actual_fps, 2),  # Add average_fps for each camera
                'timestamps': len(self.frame_timestamps[cam_name]),
                'width': width,
                'height': height,
            }

            quality_stats = self._analyze_timestamp_quality(cam_name)
            if quality_stats and quality_stats['dropped_frames'] > 0:
                print(f"[QUALITY] {cam_name}: {quality_stats['dropped_frames']} drops, {quality_stats['stability']} stability")

        # Save unified metadata for all cameras (similar to wireless version)
        if self.start_time and camera_stats:
            self._save_unified_metadata(self.start_time, duration, camera_stats)

        self.writers = {}
        self.frame_counts = {}
        self.frame_timestamps = {}
        self.recording = False
        print("[RECORDING] Video recording completed successfully")

        return camera_stats

    def _save_timestamps(self, cam_name: str) -> None:
        """Save timestamps for a specific camera to CSV file."""
        timestamps = self.frame_timestamps.get(cam_name, [])
        if not timestamps:
            return

        timestamp_df = pd.DataFrame({
            "frame_number": range(1, len(timestamps) + 1),
            "timestamp": timestamps
        })

        timestamp_file = f"{self.output_dir}/{cam_name}_timestamps.csv"
        timestamp_df.to_csv(timestamp_file, index=False, float_format='%.6f')
        print(f"[DATA] Saved {len(timestamps)} timestamps for {cam_name}")

    def write_frame(self, cam_name: str, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Thread-safe: Write a frame to the specified camera's video file."""
        with self._lock:  # Protect VideoWriter operations
            if not self.recording or cam_name not in self.writers:
                return

            try:
                self.writers[cam_name].write(frame)
                self.frame_counts[cam_name] += 1

                if timestamp is not None:
                    self.frame_timestamps[cam_name].append(timestamp)
                else:
                    self.frame_timestamps[cam_name].append(time.time())
            except Exception as e:
                print(f"[ERROR] Failed to write frame for {cam_name}: {e}")
                # Remove problematic writer to avoid continuous errors
                if cam_name in self.writers:
                    try:
                        self.writers[cam_name].release()
                    except:
                        pass
                    del self.writers[cam_name]

    def get_recording_status(self, cam_name: Optional[str] = None) -> Tuple[str, float]:
        """Get recording status information for display."""
        if not self.recording:
            return "", 0

        current_time = time.time()
        duration = current_time - self.start_time if self.start_time else 0
        current_time_str = datetime.fromtimestamp(current_time).strftime('%H:%M:%S')

        if cam_name and cam_name in self.frame_counts:
            frames = self.frame_counts[cam_name]
            return f"REC {current_time_str} ({frames} frames)", duration

        return f"REC {current_time_str}", duration

    def _analyze_timestamp_quality(self, cam_name: str) -> Dict[str, Any]:
        """Analyze timestamp quality and frame rate stability."""
        timestamps = self.frame_timestamps.get(cam_name, [])
        if len(timestamps) < 2:
            return {}

        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        avg_interval = sum(intervals) / len(intervals)
        if not hasattr(CONFIG, 'recorder') or not hasattr(CONFIG.recorder, 'fps'):
            raise ValueError("[CONFIG] Missing required 'recorder.fps'")
        target_fps = int(CONFIG.recorder.fps)
        target_interval = 1.0 / max(target_fps, 1)
        realistic_interval = 1.0 / 20

        jitter = sum(abs(interval - avg_interval) for interval in intervals) / len(intervals)
        max_jitter = max(abs(interval - avg_interval) for interval in intervals)
        dropped_frames = sum(1 for interval in intervals if interval > realistic_interval * 1.8)
        long_delays = sum(1 for interval in intervals if interval > 0.1)
        actual_fps = 1.0 / avg_interval if avg_interval > 0 else 0
        performance_ratio = actual_fps / target_fps if target_fps > 0 else 0

        return {
            "actual_fps": actual_fps,
            "target_fps": float(target_fps),
            "performance_ratio": performance_ratio,
            "avg_jitter_ms": jitter * 1000,
            "max_jitter_ms": max_jitter * 1000,
            "dropped_frames": dropped_frames,
            "frame_drop_rate": dropped_frames / len(intervals) if intervals else 0,
            "long_delays": long_delays,
            "stability": "Good" if jitter < 0.01 else "Fair" if jitter < 0.02 else "Poor"
        }


class CameraRecordingManager:
    """Manages camera recording operations and display."""

    def __init__(self, shared_state: SharedState, headless: bool = True) -> None:
        self.shared_state = shared_state
        self.fps_handler = FPSHandler()
        self.video_recorder: Optional[VideoRecorder] = None
        self.headless = True  # Force headless mode
        self.stop_event = threading.Event()
        
        # Replace single queue with multiple independent queue management
        self.workers: Dict[str, USBCamWorker] = {}  # Change to dictionary for easy access by name
        self.camera_configs: Dict[str, Dict[str, Any]] = {}
        
        self._stdin_is_tty = sys.stdin.isatty()
        self._old_term_settings: Optional[list] = None
        self._last_key_ts: float = 0.0
        self._key_debounce_sec: float = 0.3
        self._last_perf_report = time.time()
        
        if self._stdin_is_tty:
            try:
                fd = sys.stdin.fileno()
                self._old_term_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                print("[INPUT] Terminal keyboard enabled (cbreak mode)")
            except Exception as e:
                print(f"[WARNING] Failed to init terminal keyboard: {e}")
                self._stdin_is_tty = False

    def _setup_camera_windows(self) -> None:
        """Removed - running in headless mode only."""
        print("[DISPLAY] Running in headless mode - no camera windows")

    def _save_final_metadata(self) -> None:
        """Save final metadata when program exits (similar to wireless version)."""
        if not self.video_recorder or not self.video_recorder.recording:
            return
        
        current_time = time.time()
        if not self.video_recorder.start_time:
            return
            
        duration = current_time - self.video_recorder.start_time
        
        # Build camera stats for final metadata
        camera_stats = {}
        total_frames = 0
        for cam_name in self.video_recorder.frame_counts:
            frames = self.video_recorder.frame_counts[cam_name]
            fps = frames / duration if duration > 0 else 0
            camera_stats[cam_name] = {
                'frames': frames,
                'fps': fps,
                'average_fps': round(fps, 2),  # Add average_fps for each camera
                'timestamps': len(self.video_recorder.frame_timestamps.get(cam_name, [])),
                'width': self.video_recorder._last_width_height.get(cam_name, (0, 0))[0],
                'height': self.video_recorder._last_width_height.get(cam_name, (0, 0))[1],
            }
            total_frames += frames
            
        # Determine position based on task type (using gopro_name field like wireless version)
        if not hasattr(CONFIG, 'task'):
            position = "unknown"
        else:
            task_type = getattr(CONFIG.task, 'type', 'single')
            if task_type == 'single':
                position = "right"
            else:
                cam_names = list(camera_stats.keys())
                if any('left' in name.lower() for name in cam_names):
                    if any('right' in name.lower() for name in cam_names):
                        position = "bimanual"
                    else:
                        position = "left"
                else:
                    position = "right"
        
        # Save final unified metadata (using gopro_name field like wireless version does on exit)
        metadata = {
            "recording_start_time": self.video_recorder.start_time,
            "gopro_name": position,  # Use gopro_name for compatibility with wireless version
            "duration": duration,
            "num_frames": total_frames,
            "datetime": datetime.fromtimestamp(self.video_recorder.start_time).strftime(r'%Y.%m.%d_%H:%M:%S.%f'),
            "cameras": camera_stats  # Include detailed camera stats
        }
        
        metadata_filename = os.path.join(self.video_recorder.output_dir, "final_metadata.json")
        try:
            with open(metadata_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            print(f"[DATA] Saved final unified metadata: {total_frames} total frames")
        except Exception as e:
            print(f"[ERROR] Failed to save final metadata: {e}")

    def _camera_cfg(self) -> Dict[str, Dict[str, Any]]:
        """Load camera configuration from CONFIG, no default fallbacks."""
        if not hasattr(CONFIG, 'recorder'):
            raise ValueError("[CONFIG] Missing required 'recorder' section in configuration")

        recorder = CONFIG.recorder
        task_type = getattr(CONFIG.task, 'type', 'single') if hasattr(CONFIG, 'task') else 'single'
        print(f"[DEBUG] Task type: {task_type}")
        
        # Check if tactile data collection is enabled
        enable_tactile = getattr(recorder, 'enable_tactile', True)
        print(f"[CONFIG] Tactile data collection enabled: {enable_tactile}")

        if not hasattr(recorder, 'camera') or not hasattr(recorder, 'camera_paths'):
            raise ValueError("[CONFIG] Missing required 'recorder.camera' or 'recorder.camera_paths'")
        print(f"[DEBUG] Found recorder.camera and recorder.camera_paths")

        camera_cfg = getattr(recorder, 'camera')
        
        # Check new layered configuration (tactile/visual) or old unified configuration
        if hasattr(camera_cfg, 'tactile') and hasattr(camera_cfg, 'visual'):
            # New layered configuration
            tactile_cfg = camera_cfg.tactile
            visual_cfg = camera_cfg.visual
            
            if not all(hasattr(tactile_cfg, attr) for attr in ['width', 'height', 'color']):
                raise ValueError("[CONFIG] Missing required 'recorder.camera.tactile' properties (width, height, color)")
            if not all(hasattr(visual_cfg, attr) for attr in ['width', 'height', 'color']):
                raise ValueError("[CONFIG] Missing required 'recorder.camera.visual' properties (width, height, color)")
                
            # Tactile camera settings (must include exposure parameters)
            if not hasattr(tactile_cfg, 'auto_exposure') or not hasattr(tactile_cfg, 'exposure'):
                raise ValueError("[CONFIG] Missing required 'recorder.camera.tactile' exposure properties (auto_exposure, exposure)")
            
            tactile_settings = {
                'width': int(tactile_cfg.width),
                'height': int(tactile_cfg.height),
                'color': bool(tactile_cfg.color),
                'auto_exposure': int(tactile_cfg.auto_exposure),
                'exposure': int(tactile_cfg.exposure)
            }
            visual_settings = {
                'width': int(visual_cfg.width),
                'height': int(visual_cfg.height),
                'color': bool(visual_cfg.color)
            }
            
            print(f"[DEBUG] Tactile camera settings: {tactile_settings}")
            print(f"[DEBUG] Visual camera settings: {visual_settings}")
            
        else:
            # Backward compatibility: old unified configuration
            if not hasattr(camera_cfg, 'width') or not hasattr(camera_cfg, 'height') or not hasattr(camera_cfg, 'color'):
                raise ValueError("[CONFIG] Missing required 'recorder.camera' properties (width, height, color)")
            
            default_settings = {
                'width': int(camera_cfg.width),
                'height': int(camera_cfg.height),
                'color': bool(camera_cfg.color)
            }
            tactile_settings = visual_settings = default_settings
            print(f"[DEBUG] Using unified camera settings: {default_settings}")
            
        def get_camera_settings(cam_name: str) -> Dict[str, Any]:
            """Return corresponding settings based on camera name"""
            if 'visual' in cam_name.lower():
                return visual_settings
            else:
                return tactile_settings

        paths_cfg = getattr(recorder, 'camera_paths')
        # Convert OmegaConf object to standard Python dict
        paths_cfg = OmegaConf.to_container(paths_cfg, resolve=True)
        print(f"[DEBUG] Camera paths config (converted): {paths_cfg}, type: {type(paths_cfg)}")

        selected_paths = None
        if isinstance(paths_cfg, dict) and task_type in paths_cfg:
            selected_paths = paths_cfg[task_type]
            print(f"[DEBUG] Selected paths for task type '{task_type}': {selected_paths}, type: {type(selected_paths)}")
        else:
            selected_paths = paths_cfg
            print(f"[DEBUG] Using direct camera_paths: {selected_paths}, type: {type(selected_paths)}")

        cam_map: Dict[str, Dict[str, Any]] = {}
        if isinstance(selected_paths, dict):
            print(f"[DEBUG] Processing camera_paths as dict")
            for cam_name, cam_path in selected_paths.items():
                cam_name = str(cam_name)
                cam_path = str(cam_path)
                
                # Tactile data filtering logic
                if not enable_tactile and 'tactile' in cam_name.lower():
                    print(f"[CONFIG] Skipping tactile camera: {cam_name} (tactile collection disabled)")
                    continue
                
                # Get corresponding settings based on camera name
                cam_settings = get_camera_settings(cam_name)
                cam_map[cam_name] = {
                    'path': cam_path,
                    'width': cam_settings['width'],
                    'height': cam_settings['height'],
                    'color': cam_settings['color'],
                }
                #  Add exposure parameters (if exists)
                if 'auto_exposure' in cam_settings:
                    cam_map[cam_name]['auto_exposure'] = cam_settings['auto_exposure']
                if 'exposure' in cam_settings:
                    cam_map[cam_name]['exposure'] = cam_settings['exposure']
                print(f"[DEBUG] Added camera: {cam_name}, path: {cam_path}, resolution: {cam_settings['width']}x{cam_settings['height']}")
        elif isinstance(selected_paths, list):
            print(f"[DEBUG] Processing camera_paths as list")
            for item in selected_paths:
                if not isinstance(item, dict):
                    print(f"[ERROR] Invalid item in camera_paths: {item}, expected dict")
                    continue
                if 'name' in item and 'path' in item:
                    cam_name = str(item['name'])
                    cam_path = str(item['path'])
                    
                    # Tactile data filtering logic
                    if not enable_tactile and 'tactile' in cam_name.lower():
                        print(f"[CONFIG] Skipping tactile camera: {cam_name} (tactile collection disabled)")
                        continue
                    
                    # Get corresponding settings based on camera name
                    cam_settings = get_camera_settings(cam_name)
                    cam_map[cam_name] = {
                        'path': cam_path,
                        'width': cam_settings['width'],
                        'height': cam_settings['height'],
                        'color': cam_settings['color'],
                    }
                    #  Add exposure parameters (if exists)
                    if 'auto_exposure' in cam_settings:
                        cam_map[cam_name]['auto_exposure'] = cam_settings['auto_exposure']
                    if 'exposure' in cam_settings:
                        cam_map[cam_name]['exposure'] = cam_settings['exposure']
                    print(f"[DEBUG] Added camera: {cam_name}, path: {cam_path}, resolution: {cam_settings['width']}x{cam_settings['height']}")
                else:
                    print(f"[ERROR] Item missing 'name' or 'path': {item}")
                    continue
        else:
            raise ValueError(f"[CONFIG] Invalid camera_paths format: expected list or dict, got {type(selected_paths)}")

        if not cam_map:
            raise ValueError(f"[CONFIG] No valid camera configurations found in 'recorder.camera_paths' for task type '{task_type}'")

        for cam_name, props in cam_map.items():
            required_keys = ['path', 'width', 'height', 'color']
            missing_keys = [key for key in required_keys if key not in props]
            if missing_keys:
                raise ValueError(f"[CONFIG] Missing required camera properties {missing_keys} for {cam_name}")
            if not isinstance(props['width'], (int, float)) or not isinstance(props['height'], (int, float)):
                raise ValueError(f"[CONFIG] Invalid width or height for {cam_name}: must be numeric")
            if not isinstance(props['color'], bool):
                raise ValueError(f"[CONFIG] Invalid color for {cam_name}: must be boolean")
            if not props['path']:
                raise ValueError(f"[CONFIG] Invalid or empty path for {cam_name}")
            #  Display exposure parameter info
            exposure_info = ""
            if 'auto_exposure' in props and 'exposure' in props:
                exposure_info = f", auto_exposure={props['auto_exposure']}, exposure={props['exposure']}"
            print(f"[DEBUG] Validated camera: {cam_name}, props: {props}{exposure_info}")

        return cam_map   
        
    def _spawn_workers(self) -> None:
        """Spawn USB camera workers with individual queues."""
        if not hasattr(CONFIG, 'recorder') or not hasattr(CONFIG.recorder, 'fps'):
            raise ValueError("[CONFIG] Missing required 'recorder.fps'")
        
        fps = int(CONFIG.recorder.fps)
        self.camera_configs = self._camera_cfg()
        
        print(f"[WORKERS] Starting {len(self.camera_configs)} camera workers with independent queues")
        
        for cam_name, props in self.camera_configs.items():
            path = props['path']
            w = int(props['width'])
            h = int(props['height'])
            device_spec = path
            
            #  Get exposure setting parameters (tactile cameras only)
            auto_exposure = props.get('auto_exposure')
            exposure = props.get('exposure')
            
            print(f"[WORKERS] {cam_name}: auto_exposure={auto_exposure}, exposure={exposure}")
            
            # sadasd.pystyle: Independent worker thread for each camera, 30fps target
            worker = USBCamWorker(
                cam_name=cam_name,
                device=device_spec, 
                resolution=(w, h), 
                fps=30,  # Force 30fps, consistent with sadasd.py
                stop_event=self.stop_event,
                queue_size=64,  # Increase queue size to avoid blocking
                headless=True,
                auto_exposure=auto_exposure,
                exposure=exposure
            )
            
            self.workers[cam_name] = worker
            worker.start()
            print(f"[WORKERS] Started worker for {cam_name} with dedicated queue")

    def _handle_recording_state_change(self) -> None:
        """Thread-safe: Unified recording management, avoid multi-thread VideoWriter sharing"""
        if self.shared_state.is_recording():
            # Use only one VideoRecorder, manage all VideoWriters uniformly in main thread
            output_dir = self.shared_state.get_current_output_dir()
            if output_dir:
                video_dir = os.path.join(output_dir, "videos")
                os.makedirs(video_dir, exist_ok=True)
                
                # Start only one VideoRecorder instance, manage all cameras uniformly
                if self.video_recorder is None or not self.video_recorder.recording:
                    self.video_recorder = VideoRecorder(output_dir=video_dir)
        else:
            # Stop unified VideoRecorder and generate statistics
            if self.video_recorder and self.video_recorder.recording:
                camera_stats = self.video_recorder.stop_recording()
                if camera_stats:
                    self.shared_state.save_camera_stats(camera_stats)
                self.shared_state.display_summary_and_save_details()
                self.video_recorder = None

    def _process_frame(self, cam_name: str, frame: np.ndarray, timestamp: float) -> None:
        """Process a single camera frame (recording only, no display)."""
        if self.shared_state.is_recording() and self.video_recorder and not self.video_recorder.recording:
            self.video_recorder.start_recording(self.camera_configs)
        if self.video_recorder and self.video_recorder.recording:
            self.video_recorder.write_frame(cam_name, frame, timestamp)
    
    def _process_all_camera_frames(self) -> int:
        """Thread-safe: Main thread uniformly processes all camera frames"""
        processed_count = 0
        
        # Thread-safe: only read from queue, writing managed uniformly by VideoRecorder
        for cam_name, worker in self.workers.items():
            frame_data = worker.get_frame_non_blocking()
            if frame_data is not None:
                frame, timestamp = frame_data
                self._process_frame(cam_name, frame, timestamp)
                processed_count += 1
        
        return processed_count
    
    def _print_performance_report(self) -> None:
        """Periodically print performance reports"""
        current_time = time.time()
        if current_time - self._last_perf_report >= 10.0:  # Report every 10 seconds
            print("\n[PERF] === Camera Performance Report ===")
            total_captured = 0
            total_dropped = 0
            
            for cam_name, worker in self.workers.items():
                stats = worker.get_stats()
                total_captured += stats['frames_captured']
                total_dropped += stats['frames_dropped']
                queue_usage = (stats['queue_size'] / 32) * 100  # Assume queue size 32
                
                print(f"[PERF] {cam_name}: captured={stats['frames_captured']}, "
                      f"dropped={stats['frames_dropped']}, queue={queue_usage:.1f}%")
            
            if total_captured > 0:
                overall_drop_rate = (total_dropped / total_captured) * 100
                print(f"[PERF] Overall: {total_captured} frames, {overall_drop_rate:.2f}% drop rate")
            
            print("[PERF] =====================================")
            self._last_perf_report = current_time

    def _handle_key_input(self) -> bool:
        """Handle keyboard input. Returns True if should continue, False if should exit."""
        if self.shared_state.should_quit():
            return False
        if self._stdin_is_tty:
            try:
                rlist, _, _ = select.select([sys.stdin], [], [], 0)
                if rlist:
                    ch = sys.stdin.read(1)
                    now = time.time()
                    if (now - self._last_key_ts) < self._key_debounce_sec:
                        return True
                    if ch in ("q", "Q"):
                        self._last_key_ts = now
                        self.shared_state.set_exit()
                        return False
                    if ch in ("s", "S"):
                        self._last_key_ts = now
                        if self.shared_state.is_recording():
                            self.shared_state.stop_recording()
                            print("[CONTROL] Recording stopped via keyboard")
                        else:
                            self.shared_state.start_recording()
                            print("[CONTROL] Recording started via keyboard")
            except Exception as e:
                print(f"[WARNING] Terminal input failed: {e}")
                self._stdin_is_tty = False
        return True

    def run_camera_recording_thread(self) -> None:
        """Main camera recording thread function."""
        try:
            print(f"[CAMERA] Initializing USB camera recording thread (headless mode)")
            self._setup_camera_windows()
            self._spawn_workers()
            
            while not self.shared_state.should_quit():
                self._handle_recording_state_change()
                
                # Key performance optimization: batch processing, reduce loop overhead
                frames_processed_this_cycle = 0
                max_frames_per_cycle = 10  # Process maximum 10 frames per cycle
                
                for cam_name, worker in self.workers.items():
                    # Batch process all available frames for this camera
                    while frames_processed_this_cycle < max_frames_per_cycle:
                        frame_data = worker.get_frame_non_blocking()
                        if frame_data is None:
                            break
                        
                        frame, timestamp = frame_data
                        self._process_frame(cam_name, frame, timestamp)
                        frames_processed_this_cycle += 1
                
                # Sleep only when there are no frames at all
                if frames_processed_this_cycle == 0:
                    time.sleep(0.001)  # 1ms sleep
                
                if not self._handle_key_input():
                    break
            
            print("[CLEANUP] Stopping camera workers...")
            self.stop_event.set()
            
            for cam_name, worker in self.workers.items():
                worker.join(timeout=1)
                print(f"[CLEANUP] {cam_name} worker stopped")
            if self.video_recorder and self.video_recorder.recording:
                # Save final metadata when exiting
                self._save_final_metadata()
                self.video_recorder.stop_recording()
            # cv2.destroyAllWindows()  # Removed - no windows in headless mode
            if self._old_term_settings is not None and sys.stdin.isatty():
                try:
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_term_settings)
                    print("[INPUT] Terminal keyboard restored")
                except Exception as e:
                    print(f"[WARNING] Failed to restore terminal settings: {e}")
            print("[CAMERA] USB camera cleanup completed")
        except Exception as e:
            print(f"[ERROR] Camera recording thread failed: {e}")
            self.shared_state.set_exit()


def camera_recording_thread(shared_state: SharedState, headless: bool = True) -> None:
    """Thread function wrapper for camera recording."""
    manager = CameraRecordingManager(shared_state, headless=headless)
    manager.run_camera_recording_thread()


def load_config(config_path: str) -> OmegaConf:
    """Load configuration from file."""
    try:
        config = OmegaConf.load(config_path)
        print(f"[DEBUG] Loaded config: {config}")
        return config
    except Exception as e:
        print(f"[ERROR] Failed to load config file '{config_path}': {e}")
        raise


def print_instructions() -> None:
    """Print user instructions."""
    print("\n" + "="*60)
    print("           ViTaMIn-B Data Recorder (USB Cams)")
    print("="*60)
    print("Integrated directory creation functionality")
    print("RUNNING IN HEADLESS MODE")
    
    # Display tactile data collection status
    enable_tactile = getattr(CONFIG.recorder, 'enable_tactile', True) if CONFIG else True
    tactile_status = "ENABLED" if enable_tactile else "DISABLED"
    print(f"📱 TACTILE DATA COLLECTION: {tactile_status}")
    
    print("CONTROLS:")
    print("  [Q]     - Quit application")
    print("  [S/s]   - Toggle recording (start/stop)")
    print("  [Ctrl+C] - Force quit (emergency)")
    print("="*60)
    print("STATUS: Ready to record")
    print()


def create_directories_only(config_path: str) -> None:
    """
    Only create necessary directory structure, do not start recorder
    """
    global CONFIG
    
    print(f"[SETUP] Create directory structure from configuration file: {config_path}")
    CONFIG = load_config(config_path)
    
    if not hasattr(CONFIG, 'recorder') or not hasattr(CONFIG.recorder, 'output'):
        raise ValueError("[CONFIG] Missing required 'recorder.output'")
    if not hasattr(CONFIG, 'task') or not hasattr(CONFIG.task, 'name') or not hasattr(CONFIG.task, 'type'):
        raise ValueError("[CONFIG] Missing required 'task.name' or 'task.type'")
    
    base_dir = CONFIG.recorder.output
    task_name = CONFIG.task.name
    task_type = CONFIG.task.type
    
    # Create basic task directory
    task_dir = os.path.join(base_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    print(f"[SETUP]  Create task directory: {task_dir}")
    

    
    # Create other necessary directories
    demos_dir = os.path.join(task_dir, "demos")
    os.makedirs(demos_dir, exist_ok=True)
    print(f"[SETUP]  Create demos directory: {os.path.basename(demos_dir)}")
    
    all_traj_dir = os.path.join(task_dir, "all_trajectory")
    os.makedirs(all_traj_dir, exist_ok=True)
    print(f"[SETUP]  Create trajectory directory: {os.path.basename(all_traj_dir)}")
    
    print(f"[SETUP] [SUCCESS] Directory creation completed! Task: {task_name} ({task_type})")

def main() -> None:
    """Main function to run the integrated data recorder (USB cams)."""
    global CONFIG, CONFIG_FILE_PATH

    parser = argparse.ArgumentParser(description="ViTaMIn-B Integrated Data Recorder (USB cams)")
    parser.add_argument("--cfg", default="./config/task_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--mkdir-only", action="store_true",
                        help="Only create directory structure without starting recorder")
    parser.add_argument("--no-tactile", action="store_true",
                        help="Disable tactile data collection, only record visual cameras")
    parser.add_argument("--enable-tactile", action="store_true", 
                        help="Force enable tactile data collection (overrides config)")
    args = parser.parse_args()
    
    # If only creating directories, exit after execution
    if args.mkdir_only:
        try:
            create_directories_only(args.cfg)
            return
        except Exception as e:
            print(f"[ERROR] Directory creation failed: {e}")
            return

    try:
        print(f"[CONFIG] Loading configuration from: {args.cfg}")
        CONFIG_FILE_PATH = args.cfg
        CONFIG = load_config(args.cfg)

        # Apply command line arguments to override tactile settings
        if args.no_tactile:
            CONFIG.recorder.enable_tactile = False
            print("[CONFIG] Tactile data collection DISABLED via command line")
        elif args.enable_tactile:
            CONFIG.recorder.enable_tactile = True
            print("[CONFIG] Tactile data collection ENABLED via command line")
        
        # Display final tactile settings status
        tactile_enabled = getattr(CONFIG.recorder, 'enable_tactile', True)
        tactile_status = "ENABLED" if tactile_enabled else "DISABLED"
        print(f"[CONFIG] Final tactile data collection status: {tactile_status}")

        if not hasattr(CONFIG, 'recorder') or not hasattr(CONFIG.recorder, 'output'):
            raise ValueError("[CONFIG] Missing required 'recorder.output'")
        if not hasattr(CONFIG, 'task') or not hasattr(CONFIG.task, 'name') or not hasattr(CONFIG.task, 'type'):
            raise ValueError("[CONFIG] Missing required 'task.name' or 'task.type'")

        # Create basic directory structure
        base_dir = CONFIG.recorder.output
        task_name = CONFIG.task.name
        task_type = CONFIG.task.type
        
        # Create main directories
        task_dir = os.path.join(base_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        print(f"[SETUP]  Create task directory: {task_dir}")
        
        # Create subdirectories
        demos_dir = os.path.join(task_dir, "demos")
        os.makedirs(demos_dir, exist_ok=True)
        print(f"[SETUP]  Create demos directory: {os.path.basename(demos_dir)}")
        
        all_traj_dir = os.path.join(task_dir, "all_trajectory")
        os.makedirs(all_traj_dir, exist_ok=True)
        print(f"[SETUP]  Create trajectory directory: {os.path.basename(all_traj_dir)}")
        
        config_backup_dir = os.path.join(task_dir, "config_backup")
        os.makedirs(config_backup_dir, exist_ok=True)
        print(f"[SETUP]  Create configuration backup directory: {os.path.basename(config_backup_dir)}")
        
        # Immediately backup configuration file
        try:
            config_backup_path = os.path.join(config_backup_dir, os.path.basename(args.cfg))
            shutil.copy2(args.cfg, config_backup_path)
            print(f"[CONFIG]  Backup configuration file: {os.path.basename(args.cfg)}")
            
            # Save runtime configuration snapshot
            runtime_config_path = os.path.join(config_backup_dir, "runtime_config.yaml")
            with open(runtime_config_path, 'w', encoding='utf-8') as f:
                OmegaConf.save(CONFIG, f)
            print(f"[CONFIG]  Save runtime configuration snapshot: runtime_config.yaml")
            
            # Save backup information
            backup_info = {
                "backup_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "original_config_file": args.cfg,
                "task_name": task_name,
                "task_type": task_type
            }
            
            backup_info_path = os.path.join(config_backup_dir, "backup_info.json")
            with open(backup_info_path, 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, indent=4, ensure_ascii=False)
            print(f"[CONFIG]  Save backup information: backup_info.json")
            
        except Exception as e:
            print(f"[WARNING] Configuration file backup failed: {e}")

        print(f"[CONFIG] Task: {task_name} ({task_type} mode)")
        print(f"[SETUP] [SUCCESS] Directory structure creation completed!")

        shared_state = SharedState()

        # Force headless mode
        headless = True
        camera_thread = threading.Thread(target=camera_recording_thread, args=(shared_state, headless), daemon=True)
        camera_thread.start()

        print_instructions()

        while not shared_state.should_quit():
            time.sleep(0.01)  # Reduce delay, improve responsiveness

    except KeyboardInterrupt:
        print("\n[SYSTEM] Received keyboard interrupt, shutting down...")
        shared_state.set_exit()
    except Exception as e:
        print(f"[ERROR] Application failed: {e}")
        if 'shared_state' in locals():
            shared_state.set_exit()
        return

    print("[SYSTEM] Waiting for camera thread to finish...")
    camera_thread.join(timeout=2)

    #  Generate final statistics report
    shared_state.generate_final_demo_statistics()

    print("[SYSTEM] Video recording session completed successfully")
    print("="*60)


if __name__ == "__main__":
    main()