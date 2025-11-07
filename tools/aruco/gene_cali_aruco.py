import cv2
import cv2.aruco as aruco
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import tempfile
import os

def generate_aruco_marker(aruco_dict, marker_id, size_pixels=200):
    marker = aruco.generateImageMarker(aruco_dict, marker_id, size_pixels)
    return marker

def create_aruco_pdf(marker_id=10, marker_size_mm=50, output_path : str = None, target_dpi=300):

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    temp_dir = tempfile.mkdtemp()
    
    marker_size_inches = marker_size_mm / 25.4
    marker_size_pixels = int(marker_size_inches * target_dpi)
    
    print(f"Generation parameters: {marker_size_mm}mm -> {marker_size_pixels}px @ {target_dpi}DPI")
    
    marker = generate_aruco_marker(aruco_dict, marker_id, marker_size_pixels)
    
    temp_image_path = os.path.join(temp_dir, f'aruco_{marker_id}.png')
    cv2.imwrite(temp_image_path, marker)
    
    c = canvas.Canvas(output_path, pagesize=A4)
    
    page_width_mm, page_height_mm = A4[0] / mm, A4[1] / mm
    
    x = (page_width_mm - marker_size_mm) / 2
    y = (page_height_mm - marker_size_mm) / 2
    
    c.drawImage(temp_image_path, x * mm, y * mm, marker_size_mm * mm, marker_size_mm * mm)
    
    c.setLineWidth(0.1 * mm)
    
    border_mm = 1
    c.rect(
        (x - border_mm) * mm,
        (y - border_mm) * mm,
        (marker_size_mm + 2*border_mm) * mm,
        (marker_size_mm + 2*border_mm) * mm,
        stroke=1,
        fill=0
    )
    
    c.setFont("Helvetica", 10)
    c.drawString((x + marker_size_mm/3) * mm, (y - 5) * mm, f"ID: {marker_id}")
    
    c.save()
    
    os.remove(temp_image_path)
    os.rmdir(temp_dir)
    
    print(f"PDF file generated: {output_path}")
    
    verify_marker_size(marker_size_mm, marker_size_pixels, target_dpi)

def verify_marker_size(target_mm, generated_pixels, dpi):
    actual_inches = generated_pixels / dpi
    actual_mm = actual_inches * 25.4
    
    error_mm = abs(actual_mm - target_mm)
    error_percent = (error_mm / target_mm) * 100
    
    print(f"\nSize verification:")
    print(f"Target size: {target_mm:.2f}mm")
    print(f"Actual size: {actual_mm:.2f}mm")
    print(f"Error: {error_mm:.3f}mm ({error_percent:.2f}%)")
    
    if error_percent > 1.0:
        print("Warning: Size error exceeds 1%, may affect calibration accuracy!")
    else:
        print("Size accuracy is good")

if __name__ == "__main__":
    create_aruco_pdf(marker_id=35, marker_size_mm=150, output_path="aruco_marker_35_150.pdf", target_dpi=300)
