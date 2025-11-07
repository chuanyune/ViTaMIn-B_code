import cv2
import csv
import json
import argparse
from pathlib import Path

def calculate_latency(video_path: str, aruco_csv: str, video_csv: str, output_path: str):
    TARGET_ID = 30
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()

    aruco_timestamps = {}
    with open(aruco_csv, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(row['frame_id'])
                timestamp = float(row['timestamp'])
                aruco_timestamps[frame_id] = timestamp
            except ValueError as e:
                print(f"ArUco CSV data error: {row} | Error: {str(e)}")
                exit()

    video_timestamps = {}
    with open(video_csv, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(row['frame_id'])
                timestamp = float(row['timestamp'])
                video_timestamps[frame_id] = timestamp
            except ValueError as e:
                print(f"Video CSV data error: {row} | Error: {str(e)}")
                exit()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    target_frame = None
    aruco_time = None
    video_time = None
    target_frame_img = None

    print(f"Processing video, searching for ID {TARGET_ID}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if ids is not None and TARGET_ID in ids.flatten():
            target_frame = frame_count
            target_frame_img = frame.copy()
            
            if TARGET_ID in aruco_timestamps:
                aruco_time = aruco_timestamps[TARGET_ID]
            
            if frame_count in video_timestamps:
                video_time = video_timestamps[frame_count]
            
            print(f"\nFound target ID {TARGET_ID}!")
            print(f"Video frame number: {target_frame}")
            print(f"ArUco timestamp: {aruco_time}")
            print(f"Video timestamp: {video_time}")
            break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    cv2.destroyAllWindows()

    if target_frame is not None and aruco_time is not None and video_time is not None:
        latency = video_time - aruco_time
        
        video_dir = Path(video_path).parent
        
        if target_frame_img is not None:
            frame_image_path = video_dir / f"target_frame_{TARGET_ID}.jpg"
            cv2.imwrite(str(frame_image_path), target_frame_img)
            
            result_json_path = video_dir / "latency_result.json"
            
            result = {
                "target_frame": target_frame,
                "target_id": TARGET_ID,
                "aruco_timestamp": aruco_time,
                "video_timestamp": video_time,
                "latency_seconds": latency,
                "latency_ms": latency * 1000,
                "fps": fps,
                "total_frames_processed": frame_count,
                "target_frame_image": str(frame_image_path),
                "result_json_path": str(result_json_path)
            }
            
            with open(result_json_path, 'w') as f:
                json.dump(result, f, indent=4)
            
            print("\nLatency calculation completed:")
            print(f"Target frame number: {target_frame}")
            print(f"Target ID: {TARGET_ID}")
            print(f"ArUco timestamp: {aruco_time}")
            print(f"Video timestamp: {video_time}")
            print(f"Latency: {latency:.6f} seconds ({latency * 1000:.3f} milliseconds)")
            print(f"Video FPS: {fps}")
            print(f"Target frame image saved to: {frame_image_path}")
            print(f"Result JSON saved to: {result_json_path}")
        else:
            print("Cannot save target frame image")
            
            result_json_path = video_dir / "latency_result.json"
            
            result = {
                "target_frame": target_frame,
                "target_id": TARGET_ID,
                "aruco_timestamp": aruco_time,
                "video_timestamp": video_time,
                "latency_seconds": latency,
                "latency_ms": latency * 1000,
                "fps": fps,
                "total_frames_processed": frame_count
            }
            
            with open(result_json_path, 'w') as f:
                json.dump(result, f, indent=4)
    else:
        print(f"\nID {TARGET_ID} not found or missing timestamp data")
        print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    video_path = "./data_cali/tactile_2025.09.01_13.43.27.931/tactile_recording.mp4"
    aruco_csv = "./data_cali/aruco_2025.09.01_13.43.40.998/aruco_timestamps.csv"
    video_csv = "./data_cali/tactile_2025.09.01_13.43.27.931/tactile_timestamps.csv"
    output_json = "latency_result.json"
    
    print(f"Video file: {video_path}")
    print(f"ArUco CSV: {aruco_csv}")
    print(f"Video CSV: {video_csv}")
    print(f"Output JSON: {output_json}")
    
    calculate_latency(video_path, aruco_csv, video_csv, output_json)
