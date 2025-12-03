"""
Module 5-6: Business logic handlers
Combined implementation for assignments 5 and 6:
1. Theoretical derivations (documented in templates)
2. Real-time object tracking (client-side with OpenCV.js):
   (i) Marker-based (ArUco/QR code)
   (ii) Markerless tracking
   (iii) SAM2 segmentation-based
"""

import cv2
import numpy as np
import base64
import io
from core.utils import decode_base64_image, encode_image_to_base64

def compute_motion_estimate_handler(image1_data, image2_data):
    """
    Problem 1(a): Compute motion function estimates between two consecutive frames
    
    This implements optical flow to estimate motion between frames.
    
    Args:
        image1_data: Base64 encoded first frame
        image2_data: Base64 encoded second frame
        
    Returns:
        Dictionary with motion estimates and visualization
    """
    img1 = decode_base64_image(image1_data)
    img2 = decode_base64_image(image2_data)
    
    if img1 is None or img2 is None:
        return {'success': False, 'error': 'Invalid image(s)'}
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    
    # Compute optical flow using Lucas-Kanade method
    # Detect corners in first frame
    corners = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.01, minDistance=10)
    
    if corners is None or len(corners) == 0:
        return {'success': False, 'error': 'No features detected in first frame'}
    
    # Calculate optical flow
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None)
    
    # Select good points
    good_new = next_pts[status == 1]
    good_old = corners[status == 1]
    
    if len(good_new) == 0:
        return {'success': False, 'error': 'No motion detected between frames'}
    
    # Draw motion vectors
    result = img2.copy()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        
        # Draw motion vector
        cv2.line(result, (a, b), (c, d), (0, 255, 0), 2)
        cv2.circle(result, (a, b), 5, (0, 0, 255), -1)
    
    # Compute motion statistics
    motion_vectors = good_new - good_old
    mean_motion = np.mean(motion_vectors, axis=0)
    std_motion = np.std(motion_vectors, axis=0)
    
    # Create motion field visualization
    h, w = gray1.shape
    step = 20
    y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    
    # Dense optical flow for visualization
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Draw motion field
    motion_field = img2.copy()
    for y, x in zip(y_coords, x_coords):
        if 0 <= y < h and 0 <= x < w:
            fx, fy = flow[y, x]
            cv2.arrowedLine(motion_field, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1, tipLength=0.3)
    
    return {
        'success': True,
        'frame1': encode_image_to_base64(img1),
        'frame2': encode_image_to_base64(img2),
        'motion_vectors': encode_image_to_base64(result),
        'motion_field': encode_image_to_base64(motion_field),
        'mean_motion': [float(mean_motion[0]), float(mean_motion[1])],
        'std_motion': [float(std_motion[0]), float(std_motion[1])],
        'num_features': len(good_new),
        'note': 'Motion tracking equation: I(x+u, y+v, t+1) = I(x, y, t) where (u,v) is motion vector'
    }

def create_sam2_npz_from_frame(image_data):
    """
    Create SAM2 NPZ file from a captured video frame.
    Uses boundary detection to create a mask, then generates NPZ file.
    
    Args:
        image_data: Base64 encoded image (captured frame)
        
    Returns:
        Dictionary with NPZ file data (base64 encoded) and mask preview
    """
    # Decode image first to validate
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image - could not decode base64 data'}
    
    # Import boundary detection from module3
    from modules.module3.handlers import detect_boundary_handler
    
    # Detect boundary - pass as base64 string (with or without data URL prefix)
    # detect_boundary_handler uses decode_base64_image which handles both formats
    # But if it's just raw base64, add the prefix for consistency
    if not image_data.startswith('data:'):
        image_data_with_prefix = f'data:image/jpeg;base64,{image_data}'
    else:
        image_data_with_prefix = image_data
    
    try:
        result = detect_boundary_handler(image_data_with_prefix)
    except Exception as e:
        return {'success': False, 'error': f'Boundary detection failed: {str(e)}'}
    
    if result is None:
        return {'success': False, 'error': 'Boundary detection returned None - check image format'}
    
    if not isinstance(result, dict):
        return {'success': False, 'error': f'Unexpected result type: {type(result)}'}
    
    if not result.get('success', False):
        error_msg = result.get('error', 'Failed to detect object boundary')
        return {'success': False, 'error': error_msg}
    
    # Get boundary points
    if 'boundary_points' not in result:
        return {'success': False, 'error': 'Boundary detection did not return boundary_points'}
    
    boundary_points = np.array(result['boundary_points'], dtype=np.int32)
    h, w = image.shape[:2]
    
    # Create binary mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [boundary_points], 255)
    
    # Compute centroid
    moments = cv2.moments(mask)
    if moments['m00'] == 0:
        return {'success': False, 'error': 'Could not compute centroid'}
    
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    centroid = np.array([[cx, cy]], dtype=np.float32)
    
    # Reshape mask to (1, H, W) format
    mask_reshaped = (mask > 0).astype(np.uint8).reshape(1, h, w)
    
    # Create NPZ file in memory
    npz_buffer = io.BytesIO()
    np.savez(npz_buffer, masks=mask_reshaped, centroids=centroid)
    npz_buffer.seek(0)
    
    # Encode NPZ to base64
    npz_base64 = base64.b64encode(npz_buffer.read()).decode('utf-8')
    
    # Create mask preview
    mask_preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_preview_encoded = encode_image_to_base64(mask_preview)
    
    return {
        'success': True,
        'npz_file': npz_base64,
        'mask_preview': mask_preview_encoded,
        'mask_shape': [1, h, w],
        'centroid': [float(cx), float(cy)],
        'message': 'NPZ file created successfully. Download and use it for tracking.'
    }

