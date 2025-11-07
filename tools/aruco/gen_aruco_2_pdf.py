import cv2
import cv2.aruco as aruco
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import tempfile
import os

def generate_aruco_marker(aruco_dict, marker_id, size_pixels=200):
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_pixels)
    return marker

def create_aruco_pdf(marker_id=10, marker_size_mm=15, num_markers=10):
    output_path = f"aruco_markers_{marker_id}.pdf"
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    temp_dir = tempfile.mkdtemp()
    
    marker_size_pixels = int(marker_size_mm * 8)
    marker = generate_aruco_marker(aruco_dict, marker_id, marker_size_pixels)
    
    temp_image_path = os.path.join(temp_dir, f'aruco_{marker_id}.png')
    cv2.imwrite(temp_image_path, marker)
    
    c = canvas.Canvas(output_path, pagesize=A4)
    
    page_width_mm, page_height_mm = A4[0] / mm, A4[1] / mm
    
    margin_mm = 20
    spacing_mm = 10
    border_mm = 1
    
    markers_per_row = int((page_width_mm - 2 * margin_mm + spacing_mm) / (marker_size_mm + spacing_mm))
    rows = int(num_markers / markers_per_row) + (1 if num_markers % markers_per_row else 0)
    
    for i in range(num_markers):
        row = i // markers_per_row
        col = i % markers_per_row
        
        x = margin_mm + col * (marker_size_mm + spacing_mm)
        y = page_height_mm - (margin_mm + marker_size_mm) - row * (marker_size_mm + spacing_mm)
        
        c.drawImage(temp_image_path, x * mm, y * mm, marker_size_mm * mm, marker_size_mm * mm)
        
        c.setLineWidth(0.1 * mm)
        
        c.rect(
            (x - border_mm) * mm,
            (y - border_mm) * mm,
            (marker_size_mm + 2*border_mm) * mm,
            (marker_size_mm + 2*border_mm) * mm,
            stroke=1,
            fill=0
        )
        
        c.setFont("Helvetica", 8)
        c.drawString((x + marker_size_mm/3) * mm, (y - 5) * mm, f"ID: {marker_id}")
    
    c.save()
    
    os.remove(temp_image_path)
    os.rmdir(temp_dir)
    
    print(f"PDF file generated: {output_path}")

if __name__ == "__main__":
    create_aruco_pdf(marker_id=3, marker_size_mm=20, num_markers=10)
