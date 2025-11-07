from __future__ import annotations

import collections
import time
import os
from datetime import datetime
import sys
import threading

import cv2

FPS = 30

HEADLESS_MODE = 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ or not os.environ.get('DISPLAY')
if HEADLESS_MODE:
    print("Headless environment detected, enabling headless mode")

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


def find_available_cameras():
    available_cameras = []
    print("Detecting available cameras:")
    
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
                print(f" - Camera {i}: Available")
            else:
                print(f" - Camera {i}: Cannot read")
            cap.release()
        else:
            break
    
    return available_cameras


def main():
    photo_counter = 0
    current_frame = None
    camera_index = 0
    
    def take_photo():
        nonlocal photo_counter
        
        if current_frame is None:
            print("No frame available, please try again later")
            return
        
        os.makedirs("photos", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        photo_counter += 1
        filename = f"photos/CAM_{photo_counter:04d}_{timestamp}.jpg"
        
        success = cv2.imwrite(filename, current_frame)
        
        if success:
            print(f"Photo saved: {filename}")
            print(f"   Resolution: {current_frame.shape[1]}x{current_frame.shape[0]}")
            print(f"   Total photos: {photo_counter}\n")
        else:
            print(f"Failed to save photo: {filename}")
    
    available_cameras = find_available_cameras()
    
    if not available_cameras:
        print("No available cameras found!")
        return
    
    if len(available_cameras) > 1:
        print(f"\nMultiple cameras found: {available_cameras}")
        try:
            camera_index = int(input(f"Please select camera number (default: {available_cameras[0]}): ") or available_cameras[0])
            if camera_index not in available_cameras:
                print(f"Camera {camera_index} not available, using default camera {available_cameras[0]}")
                camera_index = available_cameras[0]
        except ValueError:
            camera_index = available_cameras[0]
    else:
        camera_index = available_cameras[0]
    
    print(f"Using camera: {camera_index}")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Target resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Actual resolution: {actual_width}x{actual_height}")
    print(f"FPS: {actual_fps} FPS")
    
    if not HEADLESS_MODE:
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera", 800, 600)
    
    fps_handler = FPSHandler()
    
    print("\n" + "="*50)
    print("OpenCV Camera Photo Capture Program")
    print("="*50)
    print(f"Using camera: {camera_index}")
    print(f"Resolution: {actual_width}x{actual_height}")
    if HEADLESS_MODE:
        print("Headless mode controls:")
        print("   Press Enter - Take photo")
        print("   Type 'q' + Enter - Exit program")
    else:
        print("Controls:")
        print("   Press 's' key - Take photo")
        print("   Press 'q' key - Exit program")
    print("Photos will be saved to photos/ directory")
    print("="*50 + "\n")
    
    if HEADLESS_MODE:
        def input_handler():
            while True:
                try:
                    user_input = input().strip().lower()
                    if user_input == 'q':
                        return 'quit'
                    elif user_input == '':
                        take_photo()
                except EOFError:
                    break
        
        import threading
        input_thread = threading.Thread(target=input_handler, daemon=True)
        input_thread.start()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Cannot read camera frame")
                break
            
            current_frame = frame
            fps_handler.tick("FRAME")
            
            display_frame = frame.copy()
            fps_handler.draw_fps(display_frame, "FRAME")
            
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1]-1, display_frame.shape[0]-1), (0, 255, 0), 3)
            
            draw_text(display_frame, f"Camera: {camera_index}", (10, 30), 
                    color=(0, 255, 0), bg_color=(0, 0, 0), font_scale=0.7, thickness=2)
            draw_text(display_frame, f"Photos taken: {photo_counter}", (10, 60), 
                    color=(0, 255, 0), bg_color=(0, 0, 0), font_scale=0.7, thickness=2)
            draw_text(display_frame, f"Resolution: {actual_width}x{actual_height}", (10, 90), 
                    color=(0, 255, 0), bg_color=(0, 0, 0), font_scale=0.7, thickness=2)
            
            hint_text = "Press 's' to take photo" if not HEADLESS_MODE else "Press Enter to take photo"
            draw_text(display_frame, hint_text, (10, display_frame.shape[0] - 30), 
                    color=(255, 255, 255), bg_color=(0, 0, 0), font_scale=0.6, thickness=1)
            
            if not HEADLESS_MODE:
                cv2.imshow("Camera", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    take_photo()
            else:
                time.sleep(1.0 / FPS)
    
    except KeyboardInterrupt:
        print("\nUser interrupted program")
    
    finally:
        cap.release()
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()
        
        print("\n" + "="*50)
        print(f"Photo capture completed! Total photos taken: {photo_counter}")
        if photo_counter > 0:
            print("Photos saved in photos/ directory")
        print("Thank you for using!")
        print("="*50)


class FPSHandler:

    def __init__(self, max_ticks=100):
        self._ticks = {}
        self._maxTicks = max_ticks

    def tick(self, name):
        if name not in self._ticks:
            self._ticks[name] = collections.deque(maxlen=self._maxTicks)
        self._ticks[name].append(time.monotonic())

    def tick_fps(self, name):
        if name in self._ticks and len(self._ticks[name]) > 1:
            time_diff = self._ticks[name][-1] - self._ticks[name][0]
            return (len(self._ticks[name]) - 1) / time_diff if time_diff != 0 else 0.0
        return 0.0

    def draw_fps(self, frame, name):
        frame_fps = f"{name} FPS: {round(self.tick_fps(name), 1)}"
        draw_text(
            frame,
            frame_fps,
            (5, 15),
            color=(255, 255, 255),
            bg_color=(0, 0, 0),
        )


def draw_text(
    frame,
    text,
    org,
    color=(255, 255, 255),
    bg_color=(128, 128, 128),
    font_scale=0.5,
    thickness=1,
):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, bg_color, thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


if __name__ == "__main__":
    main()
