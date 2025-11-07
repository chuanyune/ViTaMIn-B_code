import cv2
import numpy as np
import time
from datetime import datetime
import csv
import os
from pathlib import Path

def generate_aruco_markers(frame_rate, max_markers):

    frame_delay = 1 / frame_rate
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()

    counter = 0
    is_recording = False
    filename = None
    csvfile = None
    writer = None
    output_dir = None

    print("Press 'o' to start recording ArUco sequence...")
    
    while True:
        if not is_recording:
            img = np.ones((800, 800), dtype=np.uint8) * 255
            cv2.putText(img, "Press 'o' to start recording", (200, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            img = np.ones((800, 800), dtype=np.uint8) * 255
            aruco_img = cv2.aruco.generateImageMarker(aruco_dict, counter, 500)
            
            center_x = (img.shape[1] - aruco_img.shape[1]) // 2
            center_y = (img.shape[0] - aruco_img.shape[0]) // 2
            img[center_y:center_y+aruco_img.shape[0], center_x:center_x+aruco_img.shape[1]] = aruco_img
        
        cv2.imshow('ArUco Code', img)

        key = cv2.waitKey(int(frame_delay * 1000)) & 0xFF
        if key == ord('o') and not is_recording: 
            is_recording = True
            counter = 0
            
            base_dir = Path("data_cali")
            base_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")[:-3]
            folder_name = f"aruco_{timestamp}"
            
            output_dir = base_dir / folder_name
            output_dir.mkdir(exist_ok=True)
            
            filename = output_dir / "aruco_timestamps.csv"
            csvfile = open(filename, 'w', newline='')
            writer = csv.writer(csvfile)
            writer.writerow(['frame_id', 'timestamp']) 
            print(f"Output directory created successfully: {output_dir}")
            print("Recording started...")
        
        if is_recording:
            unix_time = time.time() 

            assert writer is not None
            assert csvfile is not None

            writer.writerow([counter, unix_time])
            csvfile.flush()
            print(f"Recording ID: {counter}")
            
            if counter >= max_markers:
                print(f"Reached ID {max_markers}, auto-stopping")
                break
            counter += 1
        
        if key == ord('p'):  
            break

    if csvfile:
        csvfile.close()
    cv2.destroyAllWindows()

    if filename:
        print(f"Data saved to: {filename}")
        print(f"Total frames: {counter + 1}")

if __name__ == "__main__":
    frame_rate = 30
    max_markers = 35
    generate_aruco_markers(frame_rate, max_markers)
