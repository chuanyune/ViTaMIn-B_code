#!/usr/bin/env python3

import cv2
import os
import sys
import argparse
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple


class VideoRecorder:
    
    def __init__(self, mode: str, camera_device: str, fps: int, width: int, height: int):
        self.mode = mode
        self.camera_device = camera_device
        self.cap = None
        self.video_writer = None
        self.output_dir = None
        self.video_path = None
        self.csv_path = None
        self.frame_timestamps = []
        self.frame_count = 0
        self.fps = fps
        self.width = width
        self.height = height
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
    def setup_output_directory(self) -> bool:
        try:
            base_dir = Path("data_cali")
            base_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")[:-3]
            folder_name = f"{self.mode}_{timestamp}"
            
            self.output_dir = base_dir / folder_name
            self.output_dir.mkdir(exist_ok=True)
            
            self.video_path = self.output_dir / f"{self.mode}_recording.mp4"
            self.csv_path = self.output_dir / f"{self.mode}_timestamps.csv"
            
            print(f"Output directory created successfully: {self.output_dir}")
            print(f"Video file: {self.video_path}")
            print(f"Timestamp file: {self.csv_path}")
            
            return True
            
        except Exception as e:
            print(f"Failed to create output directory: {e}")
            return False
    
    def initialize_camera(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_device)
            
            if not self.cap.isOpened():
                print(f"Cannot open camera {self.camera_device}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized successfully:")
            print(f"   Resolution: {width}x{height}")
            print(f"   FPS: {actual_fps:.2f} fps")
            
            self.video_writer = cv2.VideoWriter(
                str(self.video_path), 
                self.fourcc, 
                self.fps, 
                (self.width, self.height)
            )
            
            if not self.video_writer.isOpened():
                print(f"Cannot create video writer")
                return False
            
            return True
            
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            return False
    
    def record_frame(self) -> Tuple[bool, any]:
        ret, frame = self.cap.read()
        
        if ret:
            timestamp = time.time()
            self.frame_timestamps.append({
                'frame_id': self.frame_count,
                'timestamp': timestamp
            })
            
            self.video_writer.write(frame)
            self.frame_count += 1
            
            return True, frame
        
        return False, None
    
    def save_timestamps_csv(self):
        try:
            with open(self.csv_path, 'w', newline='') as csvfile:
                fieldnames = ['frame_id', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for frame_data in self.frame_timestamps:
                    writer.writerow(frame_data)
            
            print(f"Timestamps saved successfully: {self.csv_path}")
            print(f"Total frames: {len(self.frame_timestamps)}")
            
        except Exception as e:
            print(f"Failed to save timestamps: {e}")
    
    def start_recording(self):
        if not self.setup_output_directory():
            return False
        
        if not self.initialize_camera():
            return False
        
        print(f"\nPress 's' to start recording {self.mode} mode video, press 's' again to stop recording")
        print("-" * 50)
        
        recording_active = False
        start_time = None
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read frame")
                    break
                
                if recording_active:
                    timestamp = time.time()
                    self.frame_timestamps.append({
                        'frame_id': self.frame_count,
                        'timestamp': timestamp
                    })
                    self.video_writer.write(frame)
                    self.frame_count += 1
                
                self.add_info_overlay(frame, recording_active)
                cv2.imshow(f'{self.mode.capitalize()} Recording', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if not recording_active:
                        recording_active = True
                        start_time = time.time()
                        print("Recording started...")
                    else:
                        recording_active = False
                        end_time = time.time()
                        duration = end_time - start_time
                        print(f"\nRecording completed!")
                        print(f"Duration: {duration:.2f} seconds")
                        print(f"Total frames: {self.frame_count}")
                        if duration > 0:
                            print(f"Average FPS: {self.frame_count/duration:.2f} fps")
                        break
            
            return True
            
        except KeyboardInterrupt:
            print(f"\nUser interrupted recording")
            return True
        
        except Exception as e:
            print(f"Error occurred during recording: {e}")
            return False
        
        finally:
            self.cleanup()
    
    def add_info_overlay(self, frame, recording_active: bool):
        info_text = f"Frame: {self.frame_count} | Mode: {self.mode.upper()}"
        status_text = "RECORDING" if recording_active else "STANDBY"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        cv2.putText(frame, info_text, (10, 30), font, font_scale, (0, 255, 0), thickness)
        
        color = (0, 255, 0) if recording_active else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 60), font, font_scale, color, thickness)
        
        timestamp_text = f"Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
        cv2.putText(frame, timestamp_text, (10, frame.shape[0] - 20), font, 0.5, (255, 255, 255), 1)
    
    def cleanup(self):
        try:
            if self.frame_timestamps:
                self.save_timestamps_csv()
            
            if self.video_writer:
                self.video_writer.release()
            
            if self.cap:
                self.cap.release()
            
            cv2.destroyAllWindows()
            
            print("Cleanup completed")
            
        except Exception as e:
            print(f"Error occurred during cleanup: {e}")


def main():
    mode = "tactile"
    device = "/dev/video0"
    fps = 30
    width = 640
    height = 480
    
    
    print(f"Recording mode: {mode}")
    print(f"Camera device: {device}")
    
    recorder = VideoRecorder(mode, device, fps, width, height)
    
    success = recorder.start_recording()
    
    if success:
        print(f"\n{mode} mode video recording completed successfully!")
        print(f"Output directory: {recorder.output_dir}")
    else:
        print(f"\n{mode} mode video recording failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
