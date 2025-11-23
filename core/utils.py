"""
Core utilities for image processing and common functions
"""

import base64
import numpy as np
import cv2

def decode_base64_image(image_data):
    """
    Decode base64 image data to OpenCV image
    
    Args:
        image_data: Base64 encoded image string (data:image/png;base64,...)
        
    Returns:
        numpy.ndarray: OpenCV image or None if failed
    """
    try:
        img_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def encode_image_to_base64(image, format='.png'):
    """
    Encode OpenCV image to base64 string
    
    Args:
        image: OpenCV image (numpy.ndarray)
        format: Image format (default: '.png')
        
    Returns:
        str: Base64 encoded image string or None if failed
    """
    try:
        _, buffer = cv2.imencode(format, image)
        img_base64 = base64.b64encode(buffer).decode()
        return f'data:image/png;base64,{img_base64}'
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def convert_mm_to_inch(mm):
    """Convert millimeters to inches"""
    return mm / 25.4

def allowed_file(filename, allowed_extensions={'png', 'jpg', 'jpeg'}):
    """Check if filename has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions
