"""
Module 3: Business logic handlers
Assignment 3 Implementation:
1. Gradient images (magnitude and angle) and Laplacian of Gaussian
2. Edge detection algorithm
3. Corner detection algorithm
4. Object boundary detection using OpenCV
5. Object segmentation using ArUco markers + SAM2 comparison
"""

import os
import cv2
import numpy as np
from core.utils import decode_base64_image, encode_image_to_base64

def compute_gradient_and_log_handler(image_data):
    """
    Problem 1: Compute gradient images and Laplacian of Gaussian
    
    Returns:
        - Gradient magnitude image
        - Gradient angle image
        - Laplacian of Gaussian filtered image
    """
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Compute gradients using Sobel operators
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Compute gradient angle (in degrees)
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    # Normalize angle to 0-255 range for visualization
    angle_normalized = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert angle to color image for better visualization
    angle_colored = cv2.applyColorMap(angle_normalized, cv2.COLORMAP_HSV)
    
    # Compute Laplacian of Gaussian (LoG)
    # First apply Gaussian blur, then Laplacian
    blurred = cv2.GaussianBlur(gray, (15, 15), 2.0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    # Convert to absolute values and normalize
    laplacian_abs = np.absolute(laplacian)
    log_normalized = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return {
        'success': True,
        'original_image': encode_image_to_base64(image),
        'gradient_magnitude': encode_image_to_base64(magnitude_normalized),
        'gradient_angle': encode_image_to_base64(angle_colored),
        'laplacian_of_gaussian': encode_image_to_base64(log_normalized)
    }

def detect_edges_handler(image_data, threshold1=50, threshold2=150):
    """
    Problem 2: Edge Detection Algorithm
    
    Simple edge detection using Canny edge detector (based on gradient)
    """
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    # Create visualization: overlay edges on original
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Make edges green
    edges_colored[edges > 0] = [0, 255, 0]
    
    # Overlay on original image
    result = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)
    
    return {
        'success': True,
        'original_image': encode_image_to_base64(image),
        'edges': encode_image_to_base64(edges),
        'edges_overlay': encode_image_to_base64(result),
        'edge_count': int(np.sum(edges > 0))
    }

def detect_corners_handler(image_data, max_corners=100, quality=0.01, min_distance=10):
    """
    Problem 2: Corner Detection Algorithm
    
    Simple corner detection using Harris corner detector
    """
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Harris corner detection
    corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    
    # Normalize and threshold
    corners_normalized = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    corners_threshold = corners > (corners.max() * quality)
    
    # Get corner coordinates
    corner_coords = np.where(corners_threshold)
    
    # Alternative: Use goodFeaturesToTrack for better corner detection
    corners_gft = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, 
                                         qualityLevel=quality, minDistance=min_distance)
    
    # Draw corners on image
    result = image.copy()
    corner_count = 0
    
    if corners_gft is not None:
        corners_gft = np.int0(corners_gft)
        for corner in corners_gft:
            x, y = corner.ravel()
            cv2.circle(result, (x, y), 5, (0, 0, 255), -1)  # Red circles
            corner_count += 1
    
    # Also draw Harris corners in blue
    result[corners_threshold] = [255, 0, 0]  # Blue for Harris corners
    
    return {
        'success': True,
        'original_image': encode_image_to_base64(image),
        'corners_visualization': encode_image_to_base64(result),
        'corner_count': corner_count,
        'harris_corners': int(np.sum(corners_threshold))
    }

def detect_boundary_handler(image_data, method='contour'):
    """
    Problem 3: Object Boundary Detection
    
    Find exact boundaries of object using OpenCV (no deep learning)
    """
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Method 1: Using Canny + Contours
    if method == 'contour':
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest contour (assuming it's the main object)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Draw boundary
            result = image.copy()
            cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 3)
            
            # Fill contour area with transparency
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            # Create overlay
            overlay = image.copy()
            overlay[mask > 0] = [0, 255, 0]  # Green fill
            result_overlay = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
            cv2.drawContours(result_overlay, [largest_contour], -1, (0, 255, 0), 3)
            
            # Get boundary points
            boundary_points = largest_contour.reshape(-1, 2).tolist()
            
            return {
                'success': True,
                'original_image': encode_image_to_base64(image),
                'boundary': encode_image_to_base64(result),
                'boundary_overlay': encode_image_to_base64(result_overlay),
                'boundary_points': boundary_points,
                'area': float(cv2.contourArea(largest_contour)),
                'perimeter': float(cv2.arcLength(largest_contour, True))
            }
        else:
            return {'success': False, 'error': 'No contours found'}
    
    # Method 2: Using Watershed
    elif method == 'watershed':
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Find sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # Create result
        result = image.copy()
        result[markers == -1] = [0, 255, 0]  # Mark boundaries in green
        
        return {
            'success': True,
            'original_image': encode_image_to_base64(image),
            'boundary': encode_image_to_base64(result),
            'boundary_overlay': encode_image_to_base64(result)
        }
    
    return {'success': False, 'error': 'Invalid method'}

def segment_with_aruco_handler(image_data):
    """
    Problem 4: Object Segmentation using ArUco Markers
    
    Detects ArUco markers on object boundary and segments the object
    """
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Initialize ArUco dictionary and detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Detect markers using compatible API
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is None or len(ids) == 0:
        return {
            'success': False,
            'error': 'No ArUco markers detected. Please ensure markers are visible in the image.'
        }
    
    # Draw detected markers
    result = image.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(result, corners, ids)
    
    # Extract marker corner points for boundary
    marker_points = []
    for i, corner_set in enumerate(corners):
        corner_set = corner_set[0]  # Get first (and only) marker
        for point in corner_set:
            marker_points.append([int(point[0]), int(point[1])])
    
    marker_points = np.array(marker_points, dtype=np.int32)
    
    # Create convex hull from marker points to get object boundary
    if len(marker_points) >= 3:
        hull = cv2.convexHull(marker_points)
        
        # Draw boundary
        cv2.drawContours(result, [hull], -1, (0, 255, 0), 3)
        
        # Create mask for segmentation
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        
        # Apply mask to original image
        segmented = image.copy()
        segmented[mask == 0] = [0, 0, 0]  # Black background
        
        # Create overlay
        overlay = image.copy()
        overlay[mask > 0] = [0, 255, 0]  # Green fill
        result_overlay = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        cv2.drawContours(result_overlay, [hull], -1, (0, 255, 0), 3)
        
        # Get boundary points
        boundary_points = hull.reshape(-1, 2).tolist()
        
        return {
            'success': True,
            'original_image': encode_image_to_base64(image),
            'markers_detected': encode_image_to_base64(result),
            'segmented': encode_image_to_base64(segmented),
            'segmented_overlay': encode_image_to_base64(result_overlay),
            'marker_count': int(len(ids)),
            'boundary_points': boundary_points,
            'area': float(cv2.contourArea(hull)),
            'perimeter': float(cv2.arcLength(hull, True))
        }
    else:
        return {
            'success': False,
            'error': f'Only {len(marker_points)} marker points found. Need at least 3 points for boundary detection.'
        }

def compare_with_sam2_handler(image_data, sam2_result_data):
    """
    Problem 5: Compare ArUco segmentation with SAM2 results
    
    Compares the segmentation from ArUco markers with SAM2 model results
    """
    image = decode_base64_image(image_data)
    sam2_result = decode_base64_image(sam2_result_data) if sam2_result_data else None
    
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    if sam2_result is None:
        return {
            'success': False,
            'error': 'SAM2 result image required for comparison'
        }
    
    # Convert to grayscale for comparison
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sam2_gray = cv2.cvtColor(sam2_result, cv2.COLOR_BGR2GRAY) if len(sam2_result.shape) == 3 else sam2_result
    
    # Resize SAM2 result to match original if needed
    if sam2_gray.shape != image_gray.shape:
        sam2_gray = cv2.resize(sam2_gray, (image_gray.shape[1], image_gray.shape[0]))
    
    # Create side-by-side comparison
    comparison = np.hstack([image, sam2_result])
    
    # Calculate similarity metrics (if both are binary masks)
    # This is a placeholder - actual comparison would depend on SAM2 output format
    similarity = 0.0
    if sam2_gray.max() > 1:  # Not a binary mask
        # Convert to binary if needed
        _, sam2_binary = cv2.threshold(sam2_gray, 127, 255, cv2.THRESH_BINARY)
    else:
        sam2_binary = sam2_gray
    
    return {
        'success': True,
        'original_image': encode_image_to_base64(image),
        'sam2_result': encode_image_to_base64(sam2_result),
        'comparison': encode_image_to_base64(comparison),
        'note': 'Upload SAM2 segmented image for comparison. Manual comparison recommended.'
    }

