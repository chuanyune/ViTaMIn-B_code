import serial
import time
import json
import os
from typing import Tuple
from abc import ABC, abstractmethod

HARD_CODE_WIDTH_RANGE = [0, 1000]

def crc16_modbus(data):
    POLYNOMIAL = 0xA001
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ POLYNOMIAL
            else:
                crc >>= 1
    return crc

class BasePGIInterface(ABC):
    def __init__(self, serial_name, timeout):
        self.ser = serial.Serial(serial_name, baudrate=115200, timeout=timeout)
        self.start()

    def start(self):
        self.ser.write(bytes.fromhex("01 06 01 00 00 01 49 F6"))
        data = self.ser.readline()
        time.sleep(1) 
        print("Begin Gripper Control!")

    def stop(self):
        self.ser.close()

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_raw_obs(self) -> Tuple[int, int]:
        data = self.ser.readline()
        self.ser.write(bytes.fromhex("01 03 02 02 00 01 24 72"))
        data = self.ser.readline()
        pos = int.from_bytes(data[3:5], byteorder='big')
        self.ser.write(bytes.fromhex("01 03 01 04 00 01 C4 37"))
        data = self.ser.readline()
        velocity = int.from_bytes(data[3:5], byteorder='big')
        return pos, velocity

    def set_raw_pos(self, pos: int):
        if pos < HARD_CODE_WIDTH_RANGE[0]:
            pos = HARD_CODE_WIDTH_RANGE[0]
        elif pos > HARD_CODE_WIDTH_RANGE[1]:
            pos = HARD_CODE_WIDTH_RANGE[1]
        
        hex_str = f"{pos:04X}"
        formatted_hex_str = f"{hex_str[:2]} {hex_str[2:]}"
        hex_data_original = "01 06 01 03 " + formatted_hex_str
        hex_data = ''.join(hex_data_original.split())
        byte_data = bytearray.fromhex(hex_data)
        result_crc = crc16_modbus(byte_data)
        pos_cmd = f"{hex_data_original} {result_crc&0xff:02X} {(result_crc&0xff00)>>8:02X}"
        self.ser.write(bytes.fromhex(pos_cmd))

    def set_raw_velocity(self, velocity: int):
        if velocity < 0:
            velocity = 0
        elif velocity > 100:
            velocity = 100

        hex_str = f"{velocity:04X}"
        formatted_hex_str = f"{hex_str[:2]} {hex_str[2:]}"
        hex_data_original = "01 06 01 04 " + formatted_hex_str
        hex_data = ''.join(hex_data_original.split())
        byte_data = bytearray.fromhex(hex_data)
        result_crc = crc16_modbus(byte_data)
        velocity_cmd = f"{hex_data_original} {result_crc&0xff:02X} {(result_crc&0xff00)>>8:02X}"
        self.ser.write(bytes.fromhex(velocity_cmd))
        self.ser.readline()

class PGIInterface(BasePGIInterface):
    def __init__(self, serial_name, timeout, calibration_file_path=None):
        self.calibration_loaded = False
        self.slope = None
        self.intercept = None
        if calibration_file_path and os.path.exists(calibration_file_path):
            self.load_calibration(calibration_file_path)
        else:
            print("Warning: Calibration file not found")
        
        super().__init__(serial_name, timeout)

    def load_calibration(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.slope = float(data['slope'])
            self.intercept = float(data['intercept'])
            self.calibration_loaded = True
            
            print(f"Calibration parameters loaded from JSON file {filepath}:")
            print(f"Slope: {self.slope:.6f}, Intercept: {self.intercept:.6f}")
            if 'calibration_time' in data:
                print(f"Calibration time: {data['calibration_time']}")
            
        except Exception as e:
            print(f"Failed to load calibration file: {e}")
            print("Using default mapping parameters")

    def gripper_to_aruco(self, gripper_width: int) -> float:
        if self.calibration_loaded:
            return float(self.slope * gripper_width + self.intercept)
        else:
            raise ValueError("Calibration file not found")

    def aruco_to_gripper(self, aruco_width: float) -> int:
        if self.calibration_loaded:
            return int((aruco_width - self.intercept) / self.slope)
        else:
            raise ValueError("Calibration file not found")

    def get_obs(self) -> Tuple[float, float]:
        pos, velocity = self.get_raw_obs()
        assert pos == int(pos), "pos must be an integer"
        aruco_width = self.gripper_to_aruco(pos)
        return float(aruco_width), velocity * 0.96 / 100

    def set_pos(self, aruco_width: float):
        gripper_width = self.aruco_to_gripper(aruco_width)
        gripper_width = gripper_width
        self.set_raw_pos(int(gripper_width))

    def set_velocity(self, vel: float):
        raw_vel = int(vel / 0.096 * 100)
        self.set_raw_velocity(raw_vel)

if __name__ == "__main__":
    calibration_path = "./assets/cali_width_result/width_calibration.json" 
    pgi = PGIInterface(
        serial_name="/dev/ttyUSB0", 
        timeout=1, 
        calibration_file_path=calibration_path
    )

    pgi.start()
    pgi.set_pos(0.12)
    width, v = pgi.get_obs()
    print(width)
