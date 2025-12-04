#!/usr/bin/env python3
"""
Backend for Task 1: Object Size Estimation using Calibrated Stereo
Provides API endpoints for stereo calibration and 3D triangulation
"""

import cv2
import numpy as np
import json
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

# Global storage for calibration data
calibration_storage = {}

def create_object_points(pattern_size, square_size):
    """Create 3D object points for chessboard"""
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp

@app.route('/api/calibrate', methods=['POST'])
def calibrate_stereo():
    """Perform stereo calibration from chessboard images"""
    try:
        data = request.json
        calibration_id = data.get('id', 'default')
        pattern_size = tuple(data['pattern_size'])  # (width, height)
        square_size = float(data['square_size'])  # in mm
        image_pairs = data['image_pairs']  # List of {left: base64, right: base64}
        
        # Prepare calibration data
        objpoints = []  # 3D points
        imgpoints_left = []
        imgpoints_right = []
        
        objp = create_object_points(pattern_size, square_size)
        
        for pair in image_pairs:
            # Decode images
            left_data = base64.b64decode(pair['left'].split(',')[1])
            right_data = base64.b64decode(pair['right'].split(',')[1])
            
            left_img = np.array(Image.open(BytesIO(left_data)))
            right_img = np.array(Image.open(BytesIO(right_data)))
            
            if len(left_img.shape) == 3:
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
            else:
                left_gray = left_img
                right_gray = right_img
            
            # Find chessboard corners
            ret_left, corners_left = cv2.findChessboardCorners(
                left_gray, pattern_size, None
            )
            ret_right, corners_right = cv2.findChessboardCorners(
                right_gray, pattern_size, None
            )
            
            if ret_left and ret_right:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left = cv2.cornerSubPix(
                    left_gray, corners_left, (11, 11), (-1, -1), criteria
                )
                corners_right = cv2.cornerSubPix(
                    right_gray, corners_right, (11, 11), (-1, -1), criteria
                )
                
                objpoints.append(objp)
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)
        
        if len(objpoints) < 3:
            return jsonify({
                'success': False,
                'error': f'Need at least 3 valid pairs. Found: {len(objpoints)}'
            }), 400
        
        # Get image size from first image
        img_shape = left_gray.shape[::-1]  # (width, height)
        
        # Perform stereo calibration (full calibration including intrinsics)
        ret, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
            cv2.stereoCalibrate(
                objpoints, imgpoints_left, imgpoints_right,
                None, None, None, None,
                img_shape,
                flags=cv2.CALIB_FIX_ASPECT_RATIO,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        
        # Compute baseline
        baseline = np.linalg.norm(T)
        
        # Store calibration data
        calibration_storage[calibration_id] = {
            'cameraMatrix1': cameraMatrix1.tolist(),
            'cameraMatrix2': cameraMatrix2.tolist(),
            'distCoeffs1': distCoeffs1.tolist(),
            'distCoeffs2': distCoeffs2.tolist(),
            'R': R.tolist(),
            'T': T.tolist(),
            'E': E.tolist(),
            'F': F.tolist(),
            'image_size': img_shape,
            'baseline': float(baseline)
        }
        
        return jsonify({
            'success': True,
            'baseline': float(baseline),
            'reprojection_error': float(ret),
            'pairs_used': len(objpoints)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/triangulate', methods=['POST'])
def triangulate_points():
    """Triangulate 3D points from stereo correspondences"""
    try:
        data = request.json
        calibration_id = data.get('id', 'default')
        left_points = np.array(data['left_points'], dtype=np.float32)
        right_points = np.array(data['right_points'], dtype=np.float32)
        
        if calibration_id not in calibration_storage:
            return jsonify({
                'success': False,
                'error': 'Calibration not found. Please calibrate first.'
            }), 400
        
        calib = calibration_storage[calibration_id]
        
        # Convert to numpy arrays
        cameraMatrix1 = np.array(calib['cameraMatrix1'])
        cameraMatrix2 = np.array(calib['cameraMatrix2'])
        distCoeffs1 = np.array(calib['distCoeffs1'])
        distCoeffs2 = np.array(calib['distCoeffs2'])
        R = np.array(calib['R'])
        T = np.array(calib['T'])
        
        # Undistort points
        left_undistorted = cv2.undistortPoints(
            left_points.reshape(-1, 1, 2),
            cameraMatrix1, distCoeffs1, P=cameraMatrix1
        )
        right_undistorted = cv2.undistortPoints(
            right_points.reshape(-1, 1, 2),
            cameraMatrix2, distCoeffs2, P=cameraMatrix2
        )
        
        # Create projection matrices
        # P1 = K1 * [I | 0]
        # P2 = K2 * [R | T]
        R1 = np.eye(3)
        T1 = np.zeros((3, 1))
        P1 = cameraMatrix1 @ np.hstack([R1, T1])
        
        R2 = R
        T2 = T.reshape(3, 1)
        P2 = cameraMatrix2 @ np.hstack([R2, T2])
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, 
                                         left_undistorted.reshape(-1, 2).T,
                                         right_undistorted.reshape(-1, 2).T)
        
        # Convert to 3D (homogeneous to 3D)
        points_3d = points_4d[:3] / points_4d[3]
        points_3d = points_3d.T  # Shape: (N, 3)
        
        # Calculate statistics
        distances = np.linalg.norm(points_3d, axis=1)
        avg_distance = float(np.mean(distances))
        min_distance = float(np.min(distances))
        max_distance = float(np.max(distances))
        
        return jsonify({
            'success': True,
            'points_3d': points_3d.tolist(),
            'avg_distance': avg_distance,
            'min_distance': min_distance,
            'max_distance': max_distance
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/measure_size', methods=['POST'])
def measure_object_size():
    """Measure object size from 3D points based on shape"""
    try:
        data = request.json
        points_3d = np.array(data['points_3d'])
        shape = data.get('shape', 'rectangular')
        units = data.get('units', 'mm')
        
        # Unit conversion
        unit_scale = {'mm': 1.0, 'cm': 0.1, 'in': 0.0393701}[units]
        
        result = {}
        
        if shape == 'rectangular':
            if len(points_3d) >= 2:
                # Calculate width from first two points
                width = np.linalg.norm(points_3d[1] - points_3d[0])
                result['width'] = float(width * unit_scale)
                
                if len(points_3d) >= 4:
                    # Calculate length from first to third point (assuming rectangular)
                    length = np.linalg.norm(points_3d[2] - points_3d[0])
                    result['length'] = float(length * unit_scale)
                    result['size'] = f"{result['width']:.2f} × {result['length']:.2f} {units}"
                else:
                    result['size'] = f"Width: {result['width']:.2f} {units}"
            else:
                return jsonify({'success': False, 'error': 'Need at least 2 points for rectangular'}), 400
                
        elif shape == 'circular':
            if len(points_3d) >= 2:
                # Calculate diameter from first two points
                diameter = np.linalg.norm(points_3d[1] - points_3d[0])
                result['diameter'] = float(diameter * unit_scale)
                result['size'] = f"Diameter: {result['diameter']:.2f} {units}"
            else:
                return jsonify({'success': False, 'error': 'Need at least 2 points for circular'}), 400
                
        elif shape == 'polygon':
            if len(points_3d) >= 2:
                # Calculate all edge lengths
                edges = []
                for i in range(len(points_3d)):
                    p1 = points_3d[i]
                    p2 = points_3d[(i + 1) % len(points_3d)]
                    edge = np.linalg.norm(p2 - p1)
                    edges.append(float(edge * unit_scale))
                result['edges'] = edges
                result['size'] = f"Edges: {', '.join([f'{e:.2f}' for e in edges])} {units}"
            else:
                return jsonify({'success': False, 'error': 'Need at least 2 points for polygon'}), 400
        else:
            return jsonify({'success': False, 'error': f'Unknown shape: {shape}'}), 400
        
        # Calculate average Z distance
        avg_z = float(np.mean(points_3d[:, 2]))
        result['avg_distance_z'] = float(avg_z * unit_scale)
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/detect_chessboard', methods=['POST'])
def detect_chessboard():
    """Detect chessboard corners in stereo image pair"""
    try:
        data = request.json
        left_data = base64.b64decode(data['left_image'].split(',')[1])
        right_data = base64.b64decode(data['right_image'].split(',')[1])
        pattern_size = tuple(data['pattern_size'])
        square_size = float(data.get('square_size', 20))
        
        left_img = np.array(Image.open(BytesIO(left_data)))
        right_img = np.array(Image.open(BytesIO(right_data)))
        
        # Resize if images are too large (speeds up processing significantly)
        max_dimension = 1920
        h, w = left_img.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            left_img = cv2.resize(left_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            right_img = cv2.resize(right_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img
        
        # Try multiple detection methods for robustness
        # Method 1: Standard detection
        ret_left, corners_left = cv2.findChessboardCorners(
            left_gray, pattern_size, None
        )
        ret_right, corners_right = cv2.findChessboardCorners(
            right_gray, pattern_size, None
        )
        
        # Method 2: If standard fails, try with adaptive threshold
        if not ret_left or not ret_right:
            # Apply adaptive thresholding to improve contrast
            left_adaptive = cv2.adaptiveThreshold(
                left_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            right_adaptive = cv2.adaptiveThreshold(
                right_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            if not ret_left:
                ret_left, corners_left = cv2.findChessboardCorners(
                    left_adaptive, pattern_size, None
                )
            if not ret_right:
                ret_right, corners_right = cv2.findChessboardCorners(
                    right_adaptive, pattern_size, None
                )
        
        # Method 3: Try with different flags if still failing
        if not ret_left or not ret_right:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            if not ret_left:
                ret_left, corners_left = cv2.findChessboardCorners(
                    left_gray, pattern_size, flags
                )
            if not ret_right:
                ret_right, corners_right = cv2.findChessboardCorners(
                    right_gray, pattern_size, flags
                )
        
        if not ret_left or not ret_right:
            # Provide more helpful error message
            error_msg = f'Chessboard not found. Left: {ret_left}, Right: {ret_right}. '
            error_msg += 'Tips: Ensure the entire chessboard is visible, well-lit, and flat. '
            error_msg += f'Pattern size should be {pattern_size[0]}x{pattern_size[1]} inner corners.'
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left = cv2.cornerSubPix(
            left_gray, corners_left, (11, 11), (-1, -1), criteria
        )
        corners_right = cv2.cornerSubPix(
            right_gray, corners_right, (11, 11), (-1, -1), criteria
        )
        
        return jsonify({
            'success': True,
            'left_corners': corners_left.reshape(-1, 2).tolist(),
            'right_corners': corners_right.reshape(-1, 2).tolist()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debug_chessboard', methods=['POST'])
def debug_chessboard():
    """Debug endpoint to check image properties"""
    try:
        data = request.json
        left_data = base64.b64decode(data['left_image'].split(',')[1])
        right_data = base64.b64decode(data['right_image'].split(',')[1])
        pattern_size = tuple(data['pattern_size'])
        
        left_img = np.array(Image.open(BytesIO(left_data)))
        right_img = np.array(Image.open(BytesIO(right_data)))
        
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img
        
        # Try detection with verbose output
        ret_left, corners_left = cv2.findChessboardCorners(
            left_gray, pattern_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
        )
        ret_right, corners_right = cv2.findChessboardCorners(
            right_gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
        )
        
        return jsonify({
            'success': True,
            'left_image_shape': left_gray.shape,
            'right_image_shape': right_gray.shape,
            'left_detected': bool(ret_left),
            'right_detected': bool(ret_right),
            'pattern_size': pattern_size,
            'left_mean_brightness': float(np.mean(left_gray)),
            'right_mean_brightness': float(np.mean(right_gray)),
            'left_contrast': float(np.std(left_gray)),
            'right_contrast': float(np.std(right_gray))
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'calibrations': len(calibration_storage)})

# Serve static HTML pages
@app.route('/')
def index():
    """Serve the main landing page"""
    return send_file('index.html')

@app.route('/task1_stereo_measurement.html')
def task1():
    """Serve Task 1 page"""
    return send_file('task1_stereo_measurement.html')

@app.route('/task3_pose_hand_tracking.html')
def task3():
    """Serve Task 3 page"""
    return send_file('task3_pose_hand_tracking.html')

if __name__ == '__main__':
    print("=" * 60)
    print("Computer Vision Assignment 7 - Server Starting")
    print("=" * 60)
    print("Main page:      http://localhost:5001/")
    print("Task 1 (Stereo): http://localhost:5001/task1_stereo_measurement.html")
    print("Task 3 (Pose):   http://localhost:5001/task3_pose_hand_tracking.html")
    print("Health check:   http://localhost:5001/api/health")
    print("=" * 60)
    app.run(debug=True, port=5001)

