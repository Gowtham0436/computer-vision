"""
Module 7: Business logic handlers
Assignment 7 Implementation:
1. Calibrated Stereo Size Estimation
2. Uncalibrated Stereo Size Estimation (theoretical derivation)
3. Real-time Pose Estimation and Hand Tracking (MediaPipe/OpenPose)
"""

import os
import cv2
import numpy as np
import csv
from datetime import datetime
from core.utils import decode_base64_image, encode_image_to_base64

def estimate_size_calibrated_stereo_handler(left_image_data, right_image_data, 
                                           camera_params, object_type='rectangular'):
    """
    Problem 1: Calibrated Stereo Size Estimation
    
    Estimates object size using calibrated stereo vision.
    First estimates distance (Z) using stereo, then calculates object dimensions.
    
    Args:
        left_image_data: Base64 encoded left stereo image
        right_image_data: Base64 encoded right stereo image
        camera_params: Dictionary with camera parameters (fx, fy, cx, cy, baseline)
        object_type: 'rectangular', 'circular', or 'polygon'
        
    Returns:
        Dictionary with distance estimate and object dimensions
    """
    left_img = decode_base64_image(left_image_data)
    right_img = decode_base64_image(right_image_data)
    
    if left_img is None or right_img is None:
        return {'success': False, 'error': 'Invalid image(s)'}
    
    # Convert to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if len(left_img.shape) == 3 else left_img
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if len(right_img.shape) == 3 else right_img
    
    # Stereo matching to compute disparity
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(left_gray, right_gray)
    
    # Normalize disparity for visualization
    disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disparity_colored = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
    
    # Extract camera parameters
    fx = camera_params.get('fx', 1000.0)
    fy = camera_params.get('fy', 1000.0)
    cx = camera_params.get('cx', left_img.shape[1] / 2)
    cy = camera_params.get('cy', left_img.shape[0] / 2)
    baseline = camera_params.get('baseline', 0.1)  # in meters
    
    # For object size estimation, we need to:
    # 1. Select object region in left image
    # 2. Find corresponding points in right image
    # 3. Compute disparity
    # 4. Calculate Z (distance) using: Z = (fx * baseline) / disparity
    # 5. Calculate real-world dimensions using: X = (x - cx) * Z / fx, Y = (y - cy) * Z / fy
    
    # For demonstration, we'll use the center region of the image
    h, w = left_gray.shape
    center_x, center_y = w // 2, h // 2
    roi_size = min(w, h) // 4
    
    # Extract ROI
    roi_left = left_gray[center_y - roi_size:center_y + roi_size,
                        center_x - roi_size:center_x + roi_size]
    
    # Find corresponding region in right image (simplified - in practice use feature matching)
    # For now, assume same region (this would be improved with proper stereo matching)
    roi_right = right_gray[center_y - roi_size:center_y + roi_size,
                          center_x - roi_size:center_x + roi_size]
    
    # Compute average disparity in ROI
    roi_disparity = disparity[center_y - roi_size:center_y + roi_size,
                             center_x - roi_size:center_x + roi_size]
    valid_disparity = roi_disparity[roi_disparity > 0]
    
    if len(valid_disparity) == 0:
        return {'success': False, 'error': 'No valid disparity found. Ensure stereo images are properly aligned.'}
    
    avg_disparity = np.mean(valid_disparity)
    
    # Calculate distance Z
    if avg_disparity > 0:
        Z = (fx * baseline) / avg_disparity
    else:
        return {'success': False, 'error': 'Invalid disparity value'}
    
    # Calculate object dimensions based on type
    dimensions = {}
    
    if object_type == 'rectangular':
        # Estimate width and height
        # Using ROI dimensions in pixels
        pixel_width = roi_size * 2
        pixel_height = roi_size * 2
        
        # Convert to real-world dimensions
        real_width = (pixel_width * Z) / fx
        real_height = (pixel_height * Z) / fy
        
        dimensions = {
            'width_mm': real_width * 1000,
            'height_mm': real_height * 1000,
            'width_inches': real_width * 39.3701,
            'height_inches': real_height * 39.3701
        }
        
        # Draw annotations
        annotated = left_img.copy()
        cv2.rectangle(annotated, 
                     (center_x - roi_size, center_y - roi_size),
                     (center_x + roi_size, center_y + roi_size),
                     (0, 255, 0), 3)
        cv2.putText(annotated, f'Z: {Z:.3f}m', 
                   (center_x - roi_size, center_y - roi_size - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f'W: {dimensions["width_mm"]:.1f}mm', 
                   (center_x - roi_size, center_y - roi_size + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f'H: {dimensions["height_mm"]:.1f}mm', 
                   (center_x - roi_size, center_y - roi_size + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    elif object_type == 'circular':
        # Estimate diameter
        pixel_diameter = roi_size * 2
        
        real_diameter = (pixel_diameter * Z) / fx
        
        dimensions = {
            'diameter_mm': real_diameter * 1000,
            'diameter_inches': real_diameter * 39.3701,
            'radius_mm': (real_diameter / 2) * 1000
        }
        
        # Draw annotations
        annotated = left_img.copy()
        cv2.circle(annotated, (center_x, center_y), roi_size, (0, 255, 0), 3)
        cv2.putText(annotated, f'Z: {Z:.3f}m', 
                   (center_x - roi_size, center_y - roi_size - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f'D: {dimensions["diameter_mm"]:.1f}mm', 
                   (center_x - roi_size, center_y - roi_size + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    else:  # polygon
        # Estimate all edge dimensions
        # For polygon, we'd need to detect edges/contours
        # Simplified version: estimate bounding box dimensions
        pixel_width = roi_size * 2
        pixel_height = roi_size * 2
        
        real_width = (pixel_width * Z) / fx
        real_height = (pixel_height * Z) / fy
        
        dimensions = {
            'bounding_width_mm': real_width * 1000,
            'bounding_height_mm': real_height * 1000,
            'note': 'Polygon edge detection would provide individual edge lengths'
        }
        
        annotated = left_img.copy()
        cv2.rectangle(annotated, 
                     (center_x - roi_size, center_y - roi_size),
                     (center_x + roi_size, center_y + roi_size),
                     (0, 255, 0), 3)
        cv2.putText(annotated, f'Z: {Z:.3f}m', 
                   (center_x - roi_size, center_y - roi_size - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return {
        'success': True,
        'left_image': encode_image_to_base64(left_img),
        'right_image': encode_image_to_base64(right_img),
        'disparity_map': encode_image_to_base64(disparity_colored),
        'annotated_image': encode_image_to_base64(annotated),
        'distance_z_m': float(Z),
        'average_disparity': float(avg_disparity),
        'object_type': object_type,
        'dimensions': dimensions,
        'formula': 'Z = (fx * baseline) / disparity, then X = (x - cx) * Z / fx, Y = (y - cy) * Z / fy'
    }

def estimate_pose_hand_tracking_handler(image_data, use_mediapipe=True):
    """
    Problem 3: Real-time Pose Estimation and Hand Tracking
    
    Uses MediaPipe or OpenPose for pose and hand tracking.
    Outputs visual results and CSV data.
    
    Args:
        image_data: Base64 encoded image
        use_mediapipe: If True, use MediaPipe; else use OpenPose (if available)
        
    Returns:
        Dictionary with pose/hand landmarks and visualization
    """
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    # Create a copy for annotation
    annotated = image.copy()
    
    pose_landmarks = []
    hand_landmarks = []
    
    if use_mediapipe:
        try:
            import mediapipe as mp
            
            mp_pose = mp.solutions.pose
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            
            # Initialize MediaPipe
            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5) as pose:
                
                with mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.5) as hands:
                    
                    # Convert BGR to RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Process pose
                    pose_results = pose.process(rgb_image)
                    
                    # Process hands
                    hand_results = hands.process(rgb_image)
                    
                    # Draw pose landmarks
                    if pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated,
                            pose_results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(
                                color=(0, 255, 0), thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(
                                color=(0, 255, 0), thickness=2))
                        
                        # Extract pose landmarks
                        for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                            pose_landmarks.append({
                                'id': idx,
                                'name': mp_pose.PoseLandmark(idx).name,
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z,
                                'visibility': landmark.visibility
                            })
                    
                    # Draw hand landmarks
                    if hand_results.multi_hand_landmarks:
                        for hand_idx, hand_landmark in enumerate(hand_results.multi_hand_landmarks):
                            mp_drawing.draw_landmarks(
                                annotated,
                                hand_landmark,
                                mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(
                                    color=(255, 0, 0), thickness=2, circle_radius=2),
                                connection_drawing_spec=mp_drawing.DrawingSpec(
                                    color=(255, 0, 0), thickness=2))
                            
                            # Extract hand landmarks
                            hand_data = []
                            for idx, landmark in enumerate(hand_landmark.landmark):
                                hand_data.append({
                                    'id': idx,
                                    'x': landmark.x,
                                    'y': landmark.y,
                                    'z': landmark.z
                                })
                            
                            # Determine handedness
                            handedness = 'Unknown'
                            if hand_results.multi_handedness:
                                handedness = hand_results.multi_handedness[hand_idx].classification[0].label
                            
                            hand_landmarks.append({
                                'hand_id': hand_idx,
                                'handedness': handedness,
                                'landmarks': hand_data
                            })
            
            # Generate CSV data
            csv_data = generate_pose_csv(pose_landmarks, hand_landmarks)
            
            return {
                'success': True,
                'original_image': encode_image_to_base64(image),
                'annotated_image': encode_image_to_base64(annotated),
                'pose_landmarks': pose_landmarks,
                'hand_landmarks': hand_landmarks,
                'num_pose_landmarks': len(pose_landmarks),
                'num_hands': len(hand_landmarks),
                'csv_data': csv_data,
                'method': 'MediaPipe'
            }
            
        except ImportError:
            return {
                'success': False,
                'error': 'MediaPipe not installed. Install with: pip install mediapipe'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'MediaPipe processing failed: {str(e)}'
            }
    else:
        # OpenPose implementation would go here
        # For now, return error as OpenPose requires more setup
        return {
            'success': False,
            'error': 'OpenPose requires additional setup. Please use MediaPipe option.'
        }

def generate_pose_csv(pose_landmarks, hand_landmarks):
    """
    Generate CSV data string from pose and hand landmarks
    """
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Type', 'ID', 'Name', 'X', 'Y', 'Z', 'Visibility/Handedness'])
    
    # Write pose landmarks
    for landmark in pose_landmarks:
        writer.writerow([
            'POSE',
            landmark['id'],
            landmark['name'],
            f"{landmark['x']:.6f}",
            f"{landmark['y']:.6f}",
            f"{landmark['z']:.6f}",
            f"{landmark['visibility']:.6f}"
        ])
    
    # Write hand landmarks
    for hand in hand_landmarks:
        for landmark in hand['landmarks']:
            writer.writerow([
                f"HAND_{hand['handedness']}",
                landmark['id'],
                f"Hand_{landmark['id']}",
                f"{landmark['x']:.6f}",
                f"{landmark['y']:.6f}",
                f"{landmark['z']:.6f}",
                hand['handedness']
            ])
    
    return output.getvalue()

def save_pose_csv_to_file(csv_data, filename=None):
    """
    Save CSV data to file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"static/outputs/pose_data_{timestamp}.csv"
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as f:
        f.write(csv_data)
    
    return filename

