"""
Core utilities for image processing and common functions
Optimized for Railway deployment with memory and processing constraints
"""

import base64
import numpy as np
import cv2

# Maximum image dimensions for processing (to prevent memory issues on free tier)
MAX_IMAGE_WIDTH = 1200
MAX_IMAGE_HEIGHT = 1200
JPEG_QUALITY = 85  # Balance between quality and size

def resize_image_if_needed(image, max_width=MAX_IMAGE_WIDTH, max_height=MAX_IMAGE_HEIGHT):
    """
    Resize image if it exceeds maximum dimensions while maintaining aspect ratio.
    This is critical for Railway deployment to prevent memory issues.
    
    Args:
        image: OpenCV image (numpy.ndarray)
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized image (or original if already within limits)
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # Check if resize is needed
    if w <= max_width and h <= max_height:
        return image
    
    # Calculate scaling factor
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Use INTER_AREA for downscaling (best quality)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def decode_base64_image(image_data, auto_resize=True):
    """
    Decode base64 image data to OpenCV image with automatic resizing for deployment.
    
    Args:
        image_data: Base64 encoded image string (data:image/png;base64,...) or raw base64
        auto_resize: Whether to automatically resize large images (default: True)
        
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
        
        # Auto-resize for deployment to prevent memory issues
        if auto_resize:
            image = resize_image_if_needed(image)
                
        return image
    except base64.binascii.Error as e:
        print(f"Error: Invalid base64 data - {e}")
        return None
    except Exception as e:
        print(f"Error decoding image: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def encode_image_to_base64(image, format='.jpg', quality=JPEG_QUALITY):
    """
    Encode OpenCV image to base64 string with JPEG compression for smaller size.
    
    Args:
        image: OpenCV image (numpy.ndarray)
        format: Image format (default: '.jpg' for smaller size)
        quality: JPEG quality (0-100, default: 85)
        
    Returns:
        str: Base64 encoded image string (without data URL prefix) or None if failed
    """
    try:
        if image is None:
            print("Error: Cannot encode None image")
            return None
        
        # Resize before encoding if too large
        image = resize_image_if_needed(image)
        
        # Use JPEG for smaller file sizes (critical for deployment)
        if format == '.jpg' or format == '.jpeg':
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif format == '.png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]  # 0-9, higher = more compression
        else:
            encode_params = []
        
        _, buffer = cv2.imencode(format, image, encode_params)
        if buffer is None:
            print("Error: cv2.imencode returned None")
            return None
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    except Exception as e:
        print(f"Error encoding image: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def encode_image_to_base64_png(image):
    """
    Encode image as PNG (for cases where quality is critical like masks)
    """
    return encode_image_to_base64(image, format='.png')

def convert_mm_to_inch(mm):
    """Convert millimeters to inches"""
    return mm / 25.4

def allowed_file(filename, allowed_extensions={'png', 'jpg', 'jpeg'}):
    """Check if filename has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions
