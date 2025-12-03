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
        image_data: Base64 encoded image string (data:image/png;base64,...) or raw base64
        
    Returns:
        numpy.ndarray: OpenCV image or None if failed
    """
    try:
        if not image_data:
            print("Error: Empty image data")
            return None
        
        # Handle both formats: "data:image/png;base64,..." and raw base64
        if ',' in str(image_data):
            img_data = base64.b64decode(image_data.split(',')[1])
        else:
            img_data = base64.b64decode(image_data)
        
        if not img_data:
            print("Error: Decoded image data is empty")
            return None
            
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"Error: cv2.imdecode returned None - invalid image data (length: {len(img_data)})")
            # Try to decode as grayscale as fallback
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # Convert grayscale to BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                return None
                
        return image
    except base64.binascii.Error as e:
        print(f"Error: Invalid base64 data - {e}")
        return None
    except Exception as e:
        print(f"Error decoding image: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def encode_image_to_base64(image, format='.png'):
    """
    Encode OpenCV image to base64 string (returns raw base64, no data URL prefix)
    
    Args:
        image: OpenCV image (numpy.ndarray)
        format: Image format (default: '.png')
        
    Returns:
        str: Base64 encoded image string (without data URL prefix) or None if failed
    """
    try:
        if image is None:
            print("Error: Cannot encode None image")
            return None
        _, buffer = cv2.imencode(format, image)
        if buffer is None:
            print("Error: cv2.imencode returned None")
            return None
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        # Return just the base64 string (frontend will add data:image/png;base64, prefix)
        return img_base64
    except Exception as e:
        print(f"Error encoding image: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_mm_to_inch(mm):
    """Convert millimeters to inches"""
    return mm / 25.4

def allowed_file(filename, allowed_extensions={'png', 'jpg', 'jpeg'}):
    """Check if filename has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions
