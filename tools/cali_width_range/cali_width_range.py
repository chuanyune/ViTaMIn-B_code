import numpy as np
import json
from typing import Union, List
import warnings
import time


class WidthConverter:
    def __init__(self, slope: float = None, intercept: float = None):
        self.slope = slope
        self.intercept = intercept
        self.is_calibrated = slope is not None and intercept is not None
    
    def calibrate_from_data(self, gripper_widths: List[float], aruco_widths: List[float]):
        if len(gripper_widths) != len(aruco_widths):
            raise ValueError("gripper_widths and aruco_widths must have the same length")
        
        if len(gripper_widths) < 2:
            raise ValueError("At least 2 data points are required for calibration")
        
        gripper_array = np.array(gripper_widths)
        aruco_array = np.array(aruco_widths)
        
        coeffs = np.polyfit(gripper_array, aruco_array, 1)
        self.slope = coeffs[0]
        self.intercept = coeffs[1]
        self.is_calibrated = True
        
        predicted = self.slope * gripper_array + self.intercept
        r_squared = 1 - np.sum((aruco_array - predicted) ** 2) / np.sum((aruco_array - np.mean(aruco_array)) ** 2)
        
        print(f"Calibration completed:")
        print(f"Slope: {self.slope:.6f}")
        print(f"Intercept: {self.intercept:.6f}")
        print(f"Fit quality (R²): {r_squared:.6f}")
        
        return self.slope, self.intercept, r_squared
    
    def gripper_to_aruco(self, gripper_width: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
 
        if not self.is_calibrated:
            raise ValueError("Converter not calibrated yet, please call calibrate_from_data() first or set slope and intercept")
        
        if isinstance(gripper_width, (list, np.ndarray)):
            gripper_width = np.array(gripper_width)
        
        return self.slope * gripper_width + self.intercept
    
    def aruco_to_gripper(self, aruco_width: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        if not self.is_calibrated:
            raise ValueError("Converter not calibrated yet, please call calibrate_from_data() first or set slope and intercept")
        
        if isinstance(aruco_width, (list, np.ndarray)):
            aruco_width = np.array(aruco_width)
        
        return (aruco_width - self.intercept) / self.slope
    
    def save_calibration(self, filepath: str):
        if not self.is_calibrated:
            raise ValueError("Converter not calibrated yet")
        
        np.savez(filepath, slope=self.slope, intercept=self.intercept)
        print(f"Calibration parameters saved to: {filepath}")
    
    def load_calibration(self, filepath: str):
        data = np.load(filepath)
        self.slope = float(data['slope'])
        self.intercept = float(data['intercept'])
        self.is_calibrated = True
        print(f"Calibration parameters loaded from {filepath}:")
        print(f"Slope: {self.slope:.6f}, Intercept: {self.intercept:.6f}")

    def save_calibration_json(self, filepath: str):
        if not self.is_calibrated:
            raise ValueError("Converter not calibrated yet")
        
        calibration_data = {
            "slope": float(self.slope),
            "intercept": float(self.intercept),
            "calibration_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0"
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=4, ensure_ascii=False)
        print(f"Calibration parameters saved to JSON file: {filepath}")
    
    def load_calibration_json(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.slope = float(data['slope'])
        self.intercept = float(data['intercept'])
        self.is_calibrated = True
        
        print(f"Calibration parameters loaded from JSON file {filepath}:")
        print(f"Slope: {self.slope:.6f}, Intercept: {self.intercept:.6f}")
        if 'calibration_time' in data:
            print(f"Calibration time: {data['calibration_time']}")
        if 'version' in data:
            print(f"Version: {data['version']}")


def create_converter_from_data(gripper_widths: List[float], aruco_widths: List[float]) -> WidthConverter:
    converter = WidthConverter()
    converter.calibrate_from_data(gripper_widths, aruco_widths)
    return converter


if __name__ == "__main__":
    example_gripper_widths = [0, 200, 400, 600, 800, 1000]
    example_aruco_widths = [0.05342744446896867, 0.06899043027148796, 0.08455341607400725, 0.09946971260254554, 0.1145028226793132, 0.12981197752954984]  
    
    converter = create_converter_from_data(example_gripper_widths, example_aruco_widths)
    
    
    test_gripper_width = 250
    converted_aruco = converter.gripper_to_aruco(test_gripper_width)
    print(f"\nGripper width {test_gripper_width} -> ArUco width {converted_aruco:.3f}")
    
    test_aruco_width = 0.06899043027148796
    converted_gripper = converter.aruco_to_gripper(test_aruco_width)
    print(f"ArUco width {test_aruco_width} -> Gripper width {converted_gripper:.3f}")
    
    converter.save_calibration_json("width_calibration.json")
