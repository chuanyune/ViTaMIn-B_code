#!/usr/bin/env python3
"""
Quest headset pose data receiving client

This script connects to Quest's TCP server via ADB port forwarding,
receives headset pose data and processes/displays it.

Usage:
1. Ensure Quest has developer mode enabled and is connected to computer via USB
2. Execute ADB port forwarding command: adb forward tcp:7777 tcp:7777
3. Run the modified Unity application on Quest
4. Run this script:
python vitamin_b_data_collection_pipeline/00_get_pose.py --cfg ./config/task_config.yaml

Dependencies:
pip install numpy
"""

import socket
import json
import time
from datetime import datetime
import sys
import os
import argparse
from typing import Dict, Any, Optional
import signal
from omegaconf import OmegaConf

# Add the root directory to the path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
print(f"[SYSTEM] Root directory: {ROOT_DIR}")

# Global configuration
CONFIG: Optional[OmegaConf] = None


class QuestPoseClient:
    def __init__(self, host: str = "localhost", port: int = 7777):
        """
        Initialize Quest pose client
        
        Args:
            host: Server address (use localhost after ADB forwarding)
            port: Port number (default 7777)
        """
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.pose_count = 0
        self.start_time = time.time()
        
    def connect(self) -> bool:
        """
        Connect to Quest TCP server
        
        Returns:
            bool: True if connection successful, False if failed
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)  # 5 second connection timeout
            self.socket.connect((self.host, self.port))
            print(f"[SUCCESS] Successfully connected to {self.host}:{self.port}")
            return True
        except socket.timeout:
            print(f"[ERROR] Connection timeout: {self.host}:{self.port}")
            print("Please check:")
            print("1. Is Quest connected to computer via USB?")
            print("2. Did you execute ADB port forwarding: adb forward tcp:7777 tcp:7777")
            print("3. Is Unity application running on Quest?")
            return False
        except ConnectionRefusedError:
            print(f"[ERROR] Connection refused: {self.host}:{self.port}")
            print("Please check if TCP server is running on Quest")
            return False
        except Exception as e:
            print(f"[ERROR] Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from server"""
        if self.socket:
            self.socket.close()
            self.socket = None
            print("Disconnected")
    
    def parse_pose_data(self, json_str: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON format pose data
        
        Args:
            json_str: JSON string
            
        Returns:
            Dict: Parsed pose data, None if failed
        """
        try:
            data = json.loads(json_str.strip())
            
            # Validate data format (new format includes headset + both wrists)
            if not all(key in data for key in ['head_pose', 'left_wrist', 'right_wrist', 'timestamp']):
                print(f"Incomplete data format: {data}")
                return None
                
            # Validate head_pose format
            head = data['head_pose']
            if not all(key in head for key in ['position', 'rotation']):
                print(f"head_pose format error: {head}")
                return None
                
            # Validate left_wrist format  
            left = data['left_wrist']
            if not all(key in left for key in ['position', 'rotation']):
                print(f"left_wrist format error: {left}")
                return None
                
            # Validate right_wrist format
            right = data['right_wrist']
            if not all(key in right for key in ['position', 'rotation']):
                print(f"right_wrist format error: {right}")
                return None
                
            return data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            print(f"Data parsing error: {e}")
            return None
    
    def process_pose_data(self, pose_data: Dict[str, Any]):
        """
        Process pose data - display real-time reception frequency (using sliding window)
        
        Args:
            pose_data: Parsed pose data
        """
        current_time = time.time()
        self.pose_count += 1
        
        # Initialize time window queue (for calculating real-time FPS)
        if not hasattr(self, 'time_window'):
            self.time_window = []
        
        # Add current timestamp to window
        self.time_window.append(current_time)
        
        # Keep window size at 30 (calculate FPS for last 30 frames)
        window_size = 30
        if len(self.time_window) > window_size:
            self.time_window = self.time_window[-window_size:]
        
        # Calculate real-time FPS (using time window)
        if len(self.time_window) >= 2:
            window_duration = self.time_window[-1] - self.time_window[0]
            window_fps = (len(self.time_window) - 1) / window_duration if window_duration > 0 else 0
        else:
            window_fps = 0
        
        # Calculate overall average FPS (for display only)
        total_elapsed = current_time - self.start_time
        avg_fps = self.pose_count / total_elapsed if total_elapsed > 0 else 0
        
        # Update display (every 0.5 seconds)
        if not hasattr(self, 'last_display_time'):
            self.last_display_time = current_time
        
        if current_time - self.last_display_time >= 0.5:  # Update every 0.5 seconds
            self.last_display_time = current_time
            print(f"\rReal-time FPS: {window_fps:5.1f}Hz | Total packets: {self.pose_count:06d} | Average: {avg_fps:5.1f}Hz", end='', flush=True)
        
        # More processing logic can be added here, such as:
        # - Save to file
        # - Send to ROS
        # - Use for robot control
        # - Data analysis etc.
        
    def save_to_file(self, pose_data: Dict[str, Any], save_dir: str = None):
        """
        Save pose data to JSON file, batch saving to avoid memory issues
        
        Args:
            pose_data: Pose data
            save_dir: Save directory path, if None saves in current directory
        """
        # Initialize save-related variables
        if not hasattr(self, '_save_init'):
            self._save_init = True
            self._save_dir = save_dir or "."
            self._session_timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
            self._file_counter = 1
            self._frames_per_file = 1000  # Save 1000 frames per file
            self._current_batch = []
            self._total_saved_frames = 0
            
            # Ensure save directory exists
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            print(f"\nStarting batch data saving, {self._frames_per_file} frames per file")
        
        try:
            # Add timestamp to data
            current_unix_time = time.time()
            current_readable_time = datetime.fromtimestamp(current_unix_time).strftime("%Y.%m.%d_%H.%M.%S.%f")
            pose_data['timestamp_unix'] = current_unix_time
            pose_data['timestamp_readable'] = current_readable_time
            
            # Add to current batch
            self._current_batch.append(pose_data)
            
            # Check if file needs to be saved
            if len(self._current_batch) >= self._frames_per_file:
                self._save_current_batch()
            
            # Periodically display save progress
            self._total_saved_frames += 1
            if self._total_saved_frames % 100 == 0:
                current_file = self._file_counter if len(self._current_batch) == 0 else self._file_counter - 1
                print(f"\rRecorded {self._total_saved_frames} frames | Current file: #{current_file:03d} | Batch progress: {len(self._current_batch)}/{self._frames_per_file}", end='')
                
        except Exception as e:
            print(f"JSON file save error: {e}")
    
    def _save_current_batch(self):
        """Save current batch data to file"""
        if not self._current_batch:
            return
        
        # Generate filename
        filename = f"quest_poses_{self._session_timestamp}_part{self._file_counter:03d}.json"
        filepath = os.path.join(self._save_dir, filename)
        
        # Save data
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self._current_batch, f, ensure_ascii=False, indent=2)
            
            print(f"\nSaving file #{self._file_counter}: {filename} ({len(self._current_batch)} frames)")
            
            # Reset batch
            self._current_batch = []
            self._file_counter += 1
            
        except Exception as e:
            print(f"Batch file save error: {e}")
    
    def _save_final_batch(self):
        """Save last batch data (called when program ends)"""
        if hasattr(self, '_current_batch') and self._current_batch:
            print(f"\nSaving last batch data...")
            self._save_current_batch()
            
            # Generate summary info file
            summary = {
                "session_timestamp": self._session_timestamp,
                "total_frames": self._total_saved_frames,
                "total_files": self._file_counter - 1,
                "frames_per_file": self._frames_per_file,
                "session_duration": time.time() - self.start_time,
                "files": [
                    f"quest_poses_{self._session_timestamp}_part{i:03d}.json" 
                    for i in range(1, self._file_counter)
                ]
            }
            
            summary_filepath = os.path.join(self._save_dir, f"session_summary_{self._session_timestamp}.json")
            try:
                with open(summary_filepath, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                print(f"Saved session summary: session_summary_{self._session_timestamp}.json")
            except Exception as e:
                print(f"Summary file save error: {e}")
    
    def run(self, save_to_file: bool = False, save_dir: str = None):
        """
        Main run loop
        
        Args:
            save_to_file: Whether to save data to file
            save_dir: Save directory path, if None saves in current directory
        """
        if not self.connect():
            return
        
        self.running = True
        buffer = ""
        
        print("\nStarting to receive headset pose data...")
        print("Press Ctrl+C to stop")
        
        if save_to_file:
            save_location = f"Directory: {save_dir}" if save_dir else "Current directory"
            print(f"Data will be saved to {save_location} (batch-saved JSON files)")
        else:
            print("Only receiving data, not saving to file")
        
        print("-" * 120)
        
        try:
            while self.running:
                try:
                    # Receive data
                    data = self.socket.recv(1024).decode('utf-8')
                    if not data:
                        print("\n[ERROR] Connection disconnected")
                        break
                    
                    buffer += data
                    
                    # Process complete JSON lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            pose_data = self.parse_pose_data(line)
                            if pose_data:
                                self.process_pose_data(pose_data)
                                if save_to_file:
                                    self.save_to_file(pose_data, save_dir)
                    
                except socket.timeout:
                    continue
                except ConnectionResetError:
                    print("\n[ERROR] Connection reset")
                    break
                except Exception as e:
                    print(f"\n[ERROR] Data reception error: {e}")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nReceived stop signal")
        
        finally:
            self.running = False
            self.disconnect()
            
            # Save last batch data
            if save_to_file and hasattr(self, '_save_init'):
                self._save_final_batch()
            
            # Display statistics
            elapsed_time = time.time() - self.start_time
            avg_fps = self.pose_count / elapsed_time if elapsed_time > 0 else 0
            print(f"\nStatistics:")
            print(f"   Total received packets: {self.pose_count}")
            print(f"   Runtime: {elapsed_time:.1f} seconds")
            print(f"   Average frequency: {avg_fps:.1f}Hz")
            
            if save_to_file and hasattr(self, '_total_saved_frames'):
                print(f"   Files saved: {self._file_counter - 1}")
                print(f"   Frames saved: {self._total_saved_frames}")


def signal_handler(signum, frame):
    """Signal handler for graceful exit"""
    print("\n\nReceived exit signal")
    sys.exit(0)


def load_config(config_path: str) -> OmegaConf:
    """Load configuration file"""
    try:
        config = OmegaConf.load(config_path)
        print(f"[CONFIG] Loading configuration file: {config_path}")
        return config
    except Exception as e:
        print(f"[ERROR] Configuration file loading failed '{config_path}': {e}")
        raise

def get_all_traj_dir(config: OmegaConf) -> str:
    """Get all_trajectory directory path from configuration"""
    if not hasattr(config, 'recorder') or not hasattr(config.recorder, 'output'):
        raise ValueError("[CONFIG] Missing required configuration item 'recorder.output'")
    if not hasattr(config, 'task') or not hasattr(config.task, 'name'):
        raise ValueError("[CONFIG] Missing required configuration item 'task.name'")
    
    base_dir = config.recorder.output
    task_name = config.task.name
    task_dir = os.path.join(base_dir, task_name)
    all_traj_dir = os.path.join(task_dir, "all_trajectory")
    
    return all_traj_dir

def main():
    """Main function"""
    global CONFIG
    
    print("Quest Head Pose Client")
    print("=" * 50)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Quest headset pose data receiving client")
    parser.add_argument("--cfg", default="/home/drj/codehub/ViTaMIn-B/config/task_config.yaml",
                      help="Configuration file path")
    parser.add_argument("--no-save", action="store_true", help="Do not save data to file")
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Load configuration file
        CONFIG = load_config(args.cfg)
        
        # Save by default, unless --no-save is specified
        save_to_file = not args.no_save
        
        # Get save directory
        save_dir = get_all_traj_dir(CONFIG) if save_to_file else None
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"[SETUP]  Using trajectory directory: {save_dir}")
        else:
            print(f"[SETUP] No data saving mode")
        
        # Create and run client
        client = QuestPoseClient()
        client.run(save_to_file=save_to_file, save_dir=save_dir)
        
    except Exception as e:
        print(f"[ERROR] Program execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
