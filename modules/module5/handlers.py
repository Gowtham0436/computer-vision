"""
Module 5: Business logic handlers
Assignment 5-6 Implementation:
1. Theoretical derivations (documented in templates)
2. Real-time object tracking:
   (i) Marker-based (ArUco/QR code)
   (ii) Markerless tracking
   (iii) SAM2 segmentation-based
"""

import os
import cv2
import numpy as np
from core.utils import decode_base64_image, encode_image_to_base64

def track_with_marker_handler(image_data, marker_type='aruco'):
    """
    Problem 2(i): Marker-based Object Tracking
    
    Tracks objects using ArUco markers, QR codes, or AprilTags.
    
    Args:
        image_data: Base64 encoded image
        marker_type: 'aruco', 'qr', or 'apriltag'
        
    Returns:
        Dictionary with tracked object and visualization
    """
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    result = image.copy()
    
    tracking_info = {
        'markers_detected': 0,
        'marker_positions': [],
        'tracking_box': None,
        'center': None
    }
    
    if marker_type == 'aruco':
        # ArUco marker detection
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            aruco_params = cv2.aruco.DetectorParameters()
            
            if hasattr(cv2.aruco, 'ArucoDetector'):
                detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
                corners, ids, rejected = detector.detectMarkers(gray)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            
            if ids is not None and len(ids) > 0:
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(result, corners, ids)
                
                # Calculate bounding box and center for tracking
                all_corners = []
                for corner_set in corners:
                    corner_set = corner_set[0]
                    for corner in corner_set:
                        all_corners.append(corner)
                
                if all_corners:
                    all_corners = np.array(all_corners)
                    x_min = int(np.min(all_corners[:, 0]))
                    y_min = int(np.min(all_corners[:, 1]))
                    x_max = int(np.max(all_corners[:, 0]))
                    y_max = int(np.max(all_corners[:, 1]))
                    
                    # Draw tracking box
                    cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                    
                    # Calculate center
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2
                    cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    tracking_info['markers_detected'] = len(ids)
                    tracking_info['tracking_box'] = [x_min, y_min, x_max - x_min, y_max - y_min]
                    tracking_info['center'] = [int(center_x), int(center_y)]
                    tracking_info['marker_positions'] = [[int(c[0]), int(c[1])] for c in all_corners]
        except Exception as e:
            return {'success': False, 'error': f'ArUco detection failed: {str(e)}'}
    
    elif marker_type == 'qr':
        # QR code detection
        try:
            qr_detector = cv2.QRCodeDetector()
            retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(gray)
            
            if points is not None and len(points) > 0:
                for i, point_set in enumerate(points):
                    if point_set is not None and len(point_set) > 0:
                        point_set = point_set.astype(np.int32)
                        # Draw QR code outline
                        cv2.polylines(result, [point_set], True, (0, 255, 0), 3)
                        
                        # Calculate bounding box
                        x_coords = point_set[:, 0]
                        y_coords = point_set[:, 1]
                        x_min, y_min = int(np.min(x_coords)), int(np.min(y_coords))
                        x_max, y_max = int(np.max(x_coords)), int(np.max(y_coords))
                        
                        # Draw tracking box
                        cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                        
                        # Calculate center
                        center_x = (x_min + x_max) // 2
                        center_y = (y_min + y_max) // 2
                        cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)
                        
                        tracking_info['markers_detected'] += 1
                        if tracking_info['tracking_box'] is None:
                            tracking_info['tracking_box'] = [x_min, y_min, x_max - x_min, y_max - y_min]
                            tracking_info['center'] = [int(center_x), int(center_y)]
        except Exception as e:
            return {'success': False, 'error': f'QR code detection failed: {str(e)}'}
    
    if tracking_info['markers_detected'] == 0:
        return {'success': False, 'error': f'No {marker_type} markers detected in image'}
    
    return {
        'success': True,
        'original_image': encode_image_to_base64(image),
        'tracked_image': encode_image_to_base64(result),
        'marker_type': marker_type,
        'markers_detected': tracking_info['markers_detected'],
        'tracking_box': tracking_info['tracking_box'],
        'center': tracking_info['center']
    }

def track_markerless_handler(image_data, bbox=None, method='kcf'):
    """
    Problem 2(ii): Markerless Object Tracking
    
    Tracks objects without markers using various tracking algorithms.
    
    Args:
        image_data: Base64 encoded image
        bbox: Initial bounding box [x, y, width, height] (optional)
        method: Tracking method ('kcf', 'csrt', 'mosse', 'mil')
        
    Returns:
        Dictionary with tracking results
    """
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    result = image.copy()
    
    # If bbox is provided, initialize tracker
    if bbox is not None:
        # Initialize tracker based on method
        if method == 'kcf':
            tracker = cv2.TrackerKCF_create()
        elif method == 'csrt':
            tracker = cv2.TrackerCSRT_create()
        elif method == 'mosse':
            tracker = cv2.TrackerMOSSE_create()
        elif method == 'mil':
            tracker = cv2.TrackerMIL_create()
        else:
            tracker = cv2.TrackerKCF_create()
        
        # Convert bbox to tuple
        bbox_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        
        # Initialize tracker
        tracker.init(image, bbox_tuple)
        
        # Update tracker
        success, tracked_bbox = tracker.update(image)
        
        if success:
            # Draw tracked bounding box
            x, y, w, h = [int(v) for v in tracked_bbox]
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(result, f'Tracked ({method.upper()})', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw center
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)
            
            return {
                'success': True,
                'original_image': encode_image_to_base64(image),
                'tracked_image': encode_image_to_base64(result),
                'tracking_method': method,
                'tracking_box': [x, y, w, h],
                'center': [int(center_x), int(center_y)],
                'tracking_success': True
            }
        else:
            return {
                'success': False,
                'error': 'Tracking failed - object may be lost',
                'original_image': encode_image_to_base64(image)
            }
    else:
        # No bbox provided - return image for manual selection
        return {
            'success': True,
            'original_image': encode_image_to_base64(image),
            'tracked_image': encode_image_to_base64(result),
            'message': 'Please select bounding box to initialize tracker'
        }

def track_with_sam2_handler(image_data, sam2_mask_data=None):
    """
    Problem 2(iii): SAM2 Segmentation-based Tracking
    
    Tracks objects using SAM2 segmentation masks (from NPZ file).
    
    Args:
        image_data: Base64 encoded image
        sam2_mask_data: Base64 encoded mask image or NPZ data (optional)
        
    Returns:
        Dictionary with tracking results
    """
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    result = image.copy()
    
    # If mask is provided, use it for tracking
    if sam2_mask_data:
        try:
            # Try to decode as image first
            mask = decode_base64_image(sam2_mask_data)
            if mask is None:
                return {'success': False, 'error': 'Invalid mask data'}
            
            # Convert mask to grayscale if needed
            if len(mask.shape) == 3:
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask
            
            # Threshold mask
            _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Draw tracking box
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(result, 'SAM2 Tracked', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw contour
                cv2.drawContours(result, [largest_contour], -1, (255, 0, 0), 2)
                
                # Draw center
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Create overlay
                overlay = result.copy()
                overlay[binary_mask > 0] = [0, 255, 0]  # Green overlay
                result_overlay = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
                
                return {
                    'success': True,
                    'original_image': encode_image_to_base64(image),
                    'tracked_image': encode_image_to_base64(result),
                    'tracked_overlay': encode_image_to_base64(result_overlay),
                    'tracking_method': 'SAM2',
                    'tracking_box': [x, y, w, h],
                    'center': [int(center_x), int(center_y)],
                    'contour_area': float(cv2.contourArea(largest_contour))
                }
            else:
                return {'success': False, 'error': 'No valid contours found in SAM2 mask'}
        except Exception as e:
            return {'success': False, 'error': f'SAM2 tracking failed: {str(e)}'}
    else:
        # No mask provided - return image for mask upload
        return {
            'success': True,
            'original_image': encode_image_to_base64(image),
            'tracked_image': encode_image_to_base64(result),
            'message': 'Please upload SAM2 segmentation mask (image or NPZ file)'
        }

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

