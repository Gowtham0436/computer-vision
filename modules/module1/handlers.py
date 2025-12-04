"""
Module 1: Business logic handlers
Contains the actual dimension calculation algorithms
"""

import cv2
import numpy as np
from datetime import datetime
from core.utils import decode_base64_image, encode_image_to_base64, convert_mm_to_inch
from .evaluation_data import EVALUATION_DATA

def calculate_roi_dimensions(image_data, roi, params):
    """
    Calculate dimensions from ROI selection
    
    Args:
        image_data: Base64 encoded image
        roi: Dictionary with x, y, width, height
        params: Camera parameters (fx, fy, cx, cy, z)
        
    Returns:
        Dictionary with success status and measurements
    """
    # Decode image
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image data'}
    
    # Extract ROI coordinates and convert to integers
    x = int(roi['x'])
    y = int(roi['y'])
    w = int(roi['width'])
    h = int(roi['height'])
    
    # Camera parameters - automatically scaled for any image size/orientation
    # Default calibration values are for iPhone 17 Pro (4284x5712) but scale automatically
    # Works with portrait, landscape, or any resolution
    image_height, image_width = image.shape[:2]
    
    # Reference calibration size (iPhone 17 Pro default)
    calibrated_width = 4284
    calibrated_height = 5712
    
    # Calculate scaling factors
    scale_x = image_width / calibrated_width
    scale_y = image_height / calibrated_height
    
    # Scale camera parameters proportionally to match actual image size
    fx = params['fx'] * scale_x
    fy = params['fy'] * scale_y
    cx = params['cx'] * scale_x
    cy = params['cy'] * scale_y
    z = params['z']
    
    # Calculate real-world dimensions
    real_point1x = (x - cx) * z / fx
    real_point1y = (y - cy) * z / fy
    real_point2x = ((x + w) - cx) * z / fx
    real_point2y = ((y + h) - cy) * z / fy
    
    real_width_mm = abs(real_point2x - real_point1x)
    real_height_mm = abs(real_point2y - real_point1y)
    
    width_inches = convert_mm_to_inch(real_width_mm)
    height_inches = convert_mm_to_inch(real_height_mm)
    
    # Draw annotations
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(image, f"W: {real_width_mm:.2f} mm", (x, y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"H: {real_height_mm:.2f} mm", (x, y + h + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save annotated image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'static/outputs/roi_{timestamp}.png'
    cv2.imwrite(output_path, image)
    
    # Convert to base64 for display
    annotated_image = encode_image_to_base64(image)
    
    return {
        'success': True,
        'width_mm': round(real_width_mm, 2),
        'height_mm': round(real_height_mm, 2),
        'width_inches': round(width_inches, 2),
        'height_inches': round(height_inches, 2),
        'annotated_image': annotated_image,
        'output_path': output_path
    }

def calculate_points_dimensions(image_data, point1, point2, params):
    """
    Calculate dimensions from two points
    
    Args:
        image_data: Base64 encoded image
        point1: Dictionary with x, y coordinates
        point2: Dictionary with x, y coordinates
        params: Camera parameters (fx, fy, cx, cy, z)
        
    Returns:
        Dictionary with success status and measurements
    """
    # Decode image
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image data'}
    
    # Camera parameters - automatically scaled for any image size/orientation
    # Default calibration values are for iPhone 17 Pro (4284x5712) but scale automatically
    # Works with portrait, landscape, or any resolution
    image_height, image_width = image.shape[:2]
    
    # Reference calibration size (iPhone 17 Pro default)
    calibrated_width = 4284
    calibrated_height = 5712
    
    # Calculate scaling factors
    scale_x = image_width / calibrated_width
    scale_y = image_height / calibrated_height
    
    # Scale camera parameters proportionally to match actual image size
    fx = params['fx'] * scale_x
    fy = params['fy'] * scale_y
    cx = params['cx'] * scale_x
    cy = params['cy'] * scale_y
    z = params['z']
    
    # Extract point coordinates
    x1, y1 = int(point1['x']), int(point1['y'])
    x2, y2 = int(point2['x']), int(point2['y'])
    
    # Ensure correct order
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Calculate real-world dimensions
    X1 = (x1 - cx) * z / fx
    Y1 = (y1 - cy) * z / fy
    X2 = (x2 - cx) * z / fx
    Y2 = (y2 - cy) * z / fy
    
    width_mm = abs(X2 - X1)
    height_mm = abs(Y2 - Y1)
    width_inches = convert_mm_to_inch(width_mm)
    height_inches = convert_mm_to_inch(height_mm)
    
    # Draw annotations
    cv2.circle(image, (x1, y1), 10, (255, 0, 0), -1)
    cv2.circle(image, (x2, y2), 10, (255, 0, 0), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    cv2.putText(image, f"W: {width_mm:.1f}mm", (x1, y1 - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"H: {height_mm:.1f}mm", (x2 + 15, (y1 + y2) // 2), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Convert to base64
    annotated_image = encode_image_to_base64(image)
    
    return {
        'success': True,
        'width_mm': round(width_mm, 2),
        'height_mm': round(height_mm, 2),
        'width_inches': round(width_inches, 2),
        'height_inches': round(height_inches, 2),
        'annotated_image': annotated_image
    }

def calculate_evaluation_metrics():
    """
    Calculate evaluation metrics for all objects in EVALUATION_DATA
    
    Returns:
        List of dictionaries with evaluation metrics for each object
    """
    results = []
    
    for obj_data in EVALUATION_DATA:
        # Skip objects with no data
        if (obj_data['physical_width_mm'] == 0.0 and 
            obj_data['physical_height_mm'] == 0.0):
            continue
            
        physical_w = obj_data['physical_width_mm']
        physical_h = obj_data['physical_height_mm']
        measured_w = obj_data['measured_width_mm']
        measured_h = obj_data['measured_height_mm']
        
        # Calculate absolute errors
        width_error_mm = abs(measured_w - physical_w)
        height_error_mm = abs(measured_h - physical_h)
        
        # Calculate percentage errors
        width_error_pct = (width_error_mm / physical_w * 100) if physical_w > 0 else 0
        height_error_pct = (height_error_mm / physical_h * 100) if physical_h > 0 else 0
        
        # Calculate accuracy (100% - error%)
        width_accuracy = max(0, 100 - width_error_pct)
        height_accuracy = max(0, 100 - height_error_pct)
        
        # Average accuracy
        avg_accuracy = (width_accuracy + height_accuracy) / 2
        
        # Calculate average error percentage
        avg_error_pct = (width_error_pct + height_error_pct) / 2
        
        results.append({
            'object_name': obj_data['object_name'],
            'physical_width_mm': round(physical_w, 2),
            'physical_height_mm': round(physical_h, 2),
            'measured_width_mm': round(measured_w, 2),
            'measured_height_mm': round(measured_h, 2),
            'width_error_mm': round(width_error_mm, 2),
            'height_error_mm': round(height_error_mm, 2),
            'width_error_pct': round(width_error_pct, 2),
            'height_error_pct': round(height_error_pct, 2),
            'width_accuracy': round(width_accuracy, 2),
            'height_accuracy': round(height_accuracy, 2),
            'avg_accuracy': round(avg_accuracy, 2),
            'avg_error_pct': round(avg_error_pct, 2),
            'image_path': obj_data['image_path'],
            'notes': obj_data.get('notes', '')
        })
    
    return results
