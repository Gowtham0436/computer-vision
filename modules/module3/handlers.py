"""
Module 3: Business logic handlers
Assignment 3 Implementation - Advanced Algorithms:
1. Gradient images (magnitude and angle) and Laplacian of Gaussian
2. Edge detection algorithm (NMS + Hysteresis)
3. Corner detection algorithm (Custom Harris)
4. Object boundary detection using OpenCV (Advanced contour scoring)
5. Object segmentation using ArUco markers + SAM2 comparison
"""

import os
import cv2
import numpy as np
import base64
from core.utils import decode_base64_image, encode_image_to_base64

def percentile_from_8u(mat, p):
    """Calculate percentile from 8-bit unsigned matrix"""
    hist = np.bincount(mat.flatten(), minlength=256)
    target = (p / 100.0) * mat.size
    cum = 0
    for v in range(256):
        cum += hist[v]
        if cum >= target:
            return v
    return 255

def compute_gradient_and_log_handler(image_data):
    """
    Problem 1: Compute gradient images and Laplacian of Gaussian
    Advanced implementation matching JavaScript version
    
    Returns:
        - Gradient magnitude image
        - Gradient angle image (HSV color-coded)
        - Laplacian of Gaussian filtered image
    """
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # ========== Gradient Magnitude ==========
    # Apply Gaussian blur (3x3, sigma=0.8) before Sobel
    gray_mag = cv2.GaussianBlur(gray, (3, 3), 0.8)
    grad_x_mag = cv2.Sobel(gray_mag, cv2.CV_64F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y_mag = cv2.Sobel(gray_mag, cv2.CV_64F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    magnitude = np.sqrt(grad_x_mag**2 + grad_y_mag**2)
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # ========== Gradient Angle (HSV color-coded) ==========
    # Apply Gaussian blur (3x3, sigma=0.8)
    gray_ang = cv2.GaussianBlur(gray, (3, 3), 0.8)
    grad_x_ang = cv2.Sobel(gray_ang, cv2.CV_64F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y_ang = cv2.Sobel(gray_ang, cv2.CV_64F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    
    # Compute magnitude and angle using cartToPolar equivalent
    magnitude_ang, angle_deg = cv2.cartToPolar(grad_x_ang, grad_y_ang, angleInDegrees=True)
    
    # Normalize magnitude
    mag_normalized = cv2.normalize(magnitude_ang, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create mask from magnitude threshold
    _, mask = cv2.threshold(mag_normalized, 20, 255, cv2.THRESH_BINARY)
    
    # Convert angle to 8-bit (0-179 range for HSV hue)
    # angle in degrees: -180 to 180, convert to 0-179
    angle_normalized = (angle_deg * 0.5).astype(np.uint8)
    angle_normalized = np.clip(angle_normalized, 0, 179)
    
    # Create HSV image: H=angle, S=255, V=magnitude*mask
    hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = angle_normalized  # Hue
    hsv[:, :, 1] = 255  # Saturation
    hsv[:, :, 2] = cv2.bitwise_and(mag_normalized, mask)  # Value
    
    # Convert HSV to RGB for display
    angle_colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # ========== Laplacian of Gaussian ==========
    # Gaussian blur (5x5, sigma=1.4) then Laplacian - standard LoG filter
    blurred_log = cv2.GaussianBlur(gray, (5, 5), 1.4, borderType=cv2.BORDER_DEFAULT)
    laplacian = cv2.Laplacian(blurred_log, cv2.CV_64F, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    
    # Take absolute value and normalize to full 0-255 range for visibility
    laplacian_abs = np.abs(laplacian)
    # Enhance contrast by normalizing to full range
    laplacian_normalized = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply histogram equalization for better visibility
    laplacian_enhanced = cv2.equalizeHist(laplacian_normalized)
    
    # Ensure all images are valid before encoding
    original_encoded = encode_image_to_base64(image)
    magnitude_encoded = encode_image_to_base64(magnitude_normalized)
    angle_encoded = encode_image_to_base64(angle_colored)
    log_encoded = encode_image_to_base64(laplacian_enhanced)
    
    if not all([original_encoded, magnitude_encoded, angle_encoded, log_encoded]):
        return {'success': False, 'error': 'Failed to encode one or more result images'}
    
    return {
        'success': True,
        'original_image': original_encoded,
        'gradient_magnitude': magnitude_encoded,
        'gradient_angle': angle_encoded,
        'laplacian_of_gaussian': log_encoded
    }

def detect_edges_handler(image_data, threshold1=50, threshold2=150, edge_auto=True):
    """
    Problem 2: Edge Detection Algorithm
    Clean Canny edge detection with proper thresholding
    """
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # ========== Auto threshold using Otsu's method ==========
    if edge_auto:
        # Use Otsu's method to find optimal threshold
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Set Canny thresholds based on Otsu
        low = max(10, int(otsu_thresh * 0.5))
        high = min(255, int(otsu_thresh * 1.0))
    else:
        low = threshold1
        high = threshold2
    
    # ========== Canny Edge Detection (built-in, fast and accurate) ==========
    edges = cv2.Canny(blurred, low, high, apertureSize=3, L2gradient=True)
    
    # Optional: Clean up edges with morphological operations
    # Close small gaps
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # ========== Create output images ==========
    # 1. Binary edges (white edges on black background)
    edges_binary = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 2. Overlay (bright green edges on original for visibility)
    overlay = image.copy()
    # Make edges thicker for visibility
    thick_edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    # Apply bright green color to edge pixels
    overlay[thick_edges > 0] = [0, 255, 0]  # Bright green in BGR
    # Blend with original for better visibility
    overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    overlay[thick_edges > 0] = [0, 255, 0]  # Re-apply green on top
    
    edge_count = cv2.countNonZero(edges)
    
    # Encode images
    original_encoded = encode_image_to_base64(image)
    edges_encoded = encode_image_to_base64(edges_binary)
    overlay_encoded = encode_image_to_base64(overlay)
    
    if not all([original_encoded, edges_encoded, overlay_encoded]):
        return {'success': False, 'error': 'Failed to encode result images'}
    
    return {
        'success': True,
        'original_image': original_encoded,
        'edges': edges_encoded,
        'edges_overlay': overlay_encoded,
        'edge_count': int(edge_count)
    }

def detect_corners_handler(image_data, max_corners=500, quality=0.01, min_distance=10, k=0.04, sigma=1.5, th_rel=None, nms_size=3, top_n=None):
    """
    Problem 2: Corner Detection Algorithm
    
    Implements EXACT Harris corner detection from Assignment3:
    1. Compute gradients Ix, Iy using Sobel
    2. Compute structure tensor: Ixx, Iyy, Ixy
    3. Apply Gaussian weighting to structure tensor
    4. Compute Harris response: R = det(M) - k * trace(M)^2
    5. Apply relative threshold
    6. Non-maximum suppression via dilation
    7. Return top N corners sorted by score
    
    This is FAST (< 100ms) and ACCURATE for any image including simple shapes.
    
    Parameters (backwards compatible):
    - max_corners: Maximum number of corners to return (maps to top_n)
    - quality: Quality level for threshold (maps to th_rel)
    - min_distance: Minimum distance between corners (maps to nms_size)
    """
    # Map legacy parameters to new ones
    if top_n is None:
        top_n = max_corners
    if th_rel is None:
        th_rel = quality
    if nms_size < min_distance:
        nms_size = max(3, min_distance // 2)
    
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    rows, cols = gray.shape
    
    # Resize large images for faster processing (max 800px on longest side)
    max_dim = 800
    scale = 1.0
    if max(rows, cols) > max_dim:
        scale = max_dim / max(rows, cols)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rows, cols = gray.shape
    
    # Convert to float32 for precision
    gray32 = np.float32(gray)
    
    # ========== Step 1: Compute gradients ==========
    Ix = cv2.Sobel(gray32, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.Sobel(gray32, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    
    # ========== Step 2: Compute structure tensor components ==========
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # ========== Step 3: Apply Gaussian weighting ==========
    # Kernel size from sigma (6*sigma + 1, must be odd)
    win_k = max(3, int(round(6 * sigma + 1)))
    if win_k % 2 == 0:
        win_k += 1
    
    Sxx = cv2.GaussianBlur(Ixx, (win_k, win_k), sigma, borderType=cv2.BORDER_REPLICATE)
    Syy = cv2.GaussianBlur(Iyy, (win_k, win_k), sigma, borderType=cv2.BORDER_REPLICATE)
    Sxy = cv2.GaussianBlur(Ixy, (win_k, win_k), sigma, borderType=cv2.BORDER_REPLICATE)
    
    # ========== Step 4: Compute Harris response ==========
    # R = det(M) - k * trace(M)^2
    # det(M) = Sxx * Syy - Sxy^2
    # trace(M) = Sxx + Syy
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    R = det - k * (trace ** 2)
    
    # ========== Step 5: Apply relative threshold ==========
    max_R = R.max()
    if max_R <= 0:
        # No corners found - try with contour-based fallback
        return _fallback_corner_detection(image, gray)
    
    threshold = th_rel * max_R
    
    # ========== Step 6: Non-maximum suppression via dilation ==========
    if nms_size % 2 == 0:
        nms_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (nms_size, nms_size))
    R_dilated = cv2.dilate(R, kernel)
    
    # Find local maxima above threshold
    local_max_mask = (R == R_dilated) & (R >= threshold)
    
    # ========== Step 7: Extract corners and sort by score ==========
    coords = np.argwhere(local_max_mask)
    
    if len(coords) == 0:
        # No corners found - try fallback
        return _fallback_corner_detection(image, gray)
    
    # Get scores and sort
    scores = R[local_max_mask]
    sorted_indices = np.argsort(scores)[::-1]  # Descending
    
    # Take top N
    if top_n > 0 and len(sorted_indices) > top_n:
        sorted_indices = sorted_indices[:top_n]
    
    # Extract corner positions (scale back to original image coordinates)
    corners = []
    for idx in sorted_indices:
        y, x = coords[idx]
        score = scores[idx]
        # Scale coordinates back to original image size
        orig_x = int(x / scale) if scale != 1.0 else int(x)
        orig_y = int(y / scale) if scale != 1.0 else int(y)
        corners.append({'x': orig_x, 'y': orig_y, 'score': float(score)})
    
    # ========== Draw corners on image ==========
    result = image.copy()
    
    for corner in corners:
        x, y = corner['x'], corner['y']
        # BOLD and BRIGHT corner markers - highly visible
        # Outer bright green circle (large, thick)
        cv2.circle(result, (x, y), 12, (0, 255, 0), 3, cv2.LINE_AA)
        # Middle bright red/magenta circle
        cv2.circle(result, (x, y), 8, (255, 0, 255), 2, cv2.LINE_AA)
        # Inner bright cyan filled circle
        cv2.circle(result, (x, y), 5, (255, 255, 0), -1, cv2.LINE_AA)
        # Center white dot
        cv2.circle(result, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)
        # Cross lines for extra visibility
        cv2.line(result, (x-15, y), (x+15, y), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(result, (x, y-15), (x, y+15), (0, 255, 0), 2, cv2.LINE_AA)
    
    corner_count = len(corners)
    
    # Resize large images for faster encoding (max 1200px for display)
    display_max = 1200
    orig_h, orig_w = image.shape[:2]
    if max(orig_h, orig_w) > display_max:
        display_scale = display_max / max(orig_h, orig_w)
        image_display = cv2.resize(image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
        result_display = cv2.resize(result, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
    else:
        image_display = image
        result_display = result
    
    # Encode images (use JPEG for speed with large images)
    if max(orig_h, orig_w) > 1000:
        # Use JPEG for large images (much faster)
        _, orig_buffer = cv2.imencode('.jpg', image_display, [cv2.IMWRITE_JPEG_QUALITY, 90])
        _, result_buffer = cv2.imencode('.jpg', result_display, [cv2.IMWRITE_JPEG_QUALITY, 90])
        original_encoded = base64.b64encode(orig_buffer).decode('utf-8')
        corners_encoded = base64.b64encode(result_buffer).decode('utf-8')
    else:
        original_encoded = encode_image_to_base64(image_display)
        corners_encoded = encode_image_to_base64(result_display)
    
    if not all([original_encoded, corners_encoded]):
        return {'success': False, 'error': 'Failed to encode result images'}
    
    return {
        'success': True,
        'original_image': original_encoded,
        'corners_visualization': corners_encoded,
        'corner_count': corner_count,
        'harris_corners': corner_count,
        'corner_positions': [(c['x'], c['y']) for c in corners[:100]]
    }


def _fallback_corner_detection(image, gray):
    """
    Fallback corner detection using contour-based method.
    Works well for simple geometric shapes like squares.
    """
    rows, cols = gray.shape
    
    # Use multiple edge detection approaches
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges1 = cv2.Canny(blurred, 30, 100)
    
    # Also try Otsu threshold for solid shapes
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges2 = cv2.Canny(binary, 30, 100)
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    corners = []
    for contour in contours:
        if cv2.arcLength(contour, True) < 20:
            continue
        # Approximate polygon - vertices are corners
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        for point in approx:
            x, y = point[0]
            if 0 < x < cols - 1 and 0 < y < rows - 1:
                corners.append({'x': int(x), 'y': int(y), 'score': 1.0})
    
    # Remove duplicates using grid
    grid_size = 10
    corner_grid = {}
    for c in corners:
        key = (c['x'] // grid_size, c['y'] // grid_size)
        if key not in corner_grid:
            corner_grid[key] = c
    
    corners = list(corner_grid.values())
    
    # Draw corners
    result = image.copy()
    for corner in corners:
        x, y = corner['x'], corner['y']
        cv2.circle(result, (x, y), 3, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(result, (x, y), 1, (0, 0, 255), -1, cv2.LINE_AA)
    
    # Encode images
    original_encoded = encode_image_to_base64(image)
    corners_encoded = encode_image_to_base64(result)
    
    if not all([original_encoded, corners_encoded]):
        return {'success': False, 'error': 'Failed to encode result images'}
    
    return {
        'success': True,
        'original_image': original_encoded,
        'corners_visualization': corners_encoded,
        'corner_count': len(corners),
        'harris_corners': len(corners),
        'corner_positions': [(c['x'], c['y']) for c in corners[:100]]
    }

def detect_boundary_handler(image_data, method='contour', close_k=5, min_area_pct=2, eps_pct=1.5, center_r=40, edge_low=20, edge_high=60, edge_auto=True):
    """
    Problem 3: Object Boundary Detection
    
    GENERALIZED ROBUST implementation that works for:
    - Colorful objects (Rubik's cube, toys, etc.)
    - Monochrome objects (books, boxes, etc.)
    - Objects at various distances and angles
    
    Uses multiple detection strategies and intelligent scoring.
    """
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    rows, cols = image.shape[:2]
    img_area = rows * cols
    
    # ========== STEP 1: Resize for faster processing ==========
    max_dim = 800
    scale = 1.0
    if max(rows, cols) > max_dim:
        scale = max_dim / max(rows, cols)
        working_img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        working_img = image.copy()
    
    w_rows, w_cols = working_img.shape[:2]
    w_area = w_rows * w_cols
    
    # ========== STEP 2: Prepare multiple representations ==========
    gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(working_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(working_img, cv2.COLOR_BGR2LAB)
    
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    l_channel = lab[:, :, 0]
    
    # Bilateral filter preserves edges while reducing noise
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # ========== STEP 3: Collect candidate contours from multiple methods ==========
    all_contours = []
    
    # Adaptive kernel sizes based on image dimensions
    small_k = max(3, int(min(w_rows, w_cols) * 0.01))
    medium_k = max(5, int(min(w_rows, w_cols) * 0.02))
    large_k = max(15, int(min(w_rows, w_cols) * 0.04))
    
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_k, small_k))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (medium_k, medium_k))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (large_k, large_k))
        
    # ----- METHOD 1: Edge detection with LARGE morphology (merges nearby edges) -----
    # This is crucial for objects like Rubik's cube where internal edges exist
    for low, high in [(20, 60), (30, 90), (50, 150)]:
        edges = cv2.Canny(filtered, low, high)
        
        # Large dilation to merge nearby edges into one contour
        edges_dilated = cv2.dilate(edges, kernel_large, iterations=2)
        edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 0.01 * w_area < area < 0.85 * w_area:
                all_contours.append((cnt, 'edge_large', area))
    
    # ----- METHOD 2: Edge detection with SMALL morphology (preserves shape) -----
    for low, high in [(30, 90), (50, 150)]:
        edges = cv2.Canny(filtered, low, high)
        edges_dilated = cv2.dilate(edges, kernel_medium, iterations=2)
        edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
        
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 0.005 * w_area < area < 0.7 * w_area:
                all_contours.append((cnt, 'edge_small', area))
    
    # ----- METHOD 3: Color-based detection (for colorful objects) -----
    # Adaptive saturation threshold using Otsu
    _, sat_otsu = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
    # Fixed high saturation threshold
    _, sat_high = cv2.threshold(saturation, 70, 255, cv2.THRESH_BINARY)
    
    for mask in [sat_otsu, sat_high]:
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 0.01 * w_area < area < 0.7 * w_area:
                all_contours.append((cnt, 'color', area))
    
    # ----- METHOD 4: Intensity-based detection (for any object) -----
    # Adaptive thresholding works well for objects with different brightness than background
    adaptive = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 21, 5)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel_small, iterations=1)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0.01 * w_area < area < 0.7 * w_area:
            all_contours.append((cnt, 'adaptive', area))
    
    # ----- METHOD 5: Otsu thresholding on grayscale -----
    _, otsu = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0.01 * w_area < area < 0.7 * w_area:
            all_contours.append((cnt, 'otsu', area))
    
    if not all_contours:
        return {'success': False, 'error': 'No object boundary found'}
    
    # ========== STEP 4: Score and select best contour ==========
    best_contour = None
    best_score = -1
    img_cx, img_cy = w_cols / 2, w_rows / 2
    
    # Pre-compute saturation mask for color scoring
    sat_mask_50 = (saturation > 50).astype(np.uint8) * 255
    
    for cnt, method_name, area in all_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_area = w * h
        
        if bbox_area == 0:
            continue
        
        # Fill ratio
        fill_ratio = area / bbox_area
        if fill_ratio < 0.25:
            continue
        
        # Centroid
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        
        # ===== SCORING FACTORS =====
        
        # 1. Edge penalty (contours touching image borders)
        margin = max(5, int(min(w_rows, w_cols) * 0.015))
        touches_left = x <= margin
        touches_top = y <= margin
        touches_right = x + w >= w_cols - margin
        touches_bottom = y + h >= w_rows - margin
        edge_count = sum([touches_left, touches_top, touches_right, touches_bottom])
        
        if edge_count >= 3:
            edge_penalty = 0.05  # Almost certainly background
        elif edge_count == 2:
            # Check which edges - top+side is likely background
            if touches_top:
                edge_penalty = 0.15
            elif touches_bottom:
                edge_penalty = 0.6  # Bottom+side could be object with reflection
            else:
                edge_penalty = 0.3
        elif edge_count == 1:
            edge_penalty = 0.85 if touches_bottom else 0.7
        else:
            edge_penalty = 1.0
        
        # 2. Center score (prefer objects near image center)
        dist_to_center = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
        max_dist = np.sqrt(img_cx**2 + img_cy**2)
        center_score = 1.0 - (dist_to_center / max_dist) * 0.4
        
        # 3. Aspect ratio (allow wide range for generality)
        aspect = w / h if h > 0 else 1
        if 0.3 <= aspect <= 3.0:
            aspect_score = 1.0
        elif 0.15 <= aspect <= 6.0:
            aspect_score = 0.6
        else:
            aspect_score = 0.3
        
        # 4. Size score (prefer medium-sized objects, allow large ones)
        size_ratio = area / w_area
        if 0.03 <= size_ratio <= 0.45:
            size_score = 1.0
        elif 0.015 <= size_ratio <= 0.60:
            size_score = 0.75
        elif 0.005 <= size_ratio <= 0.75:
            size_score = 0.4
        else:
            size_score = 0.1
        
        # 5. Solidity (how filled the convex hull is)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        solidity_score = solidity ** 0.5  # Square root to be less harsh
        
        # 6. Color content (high saturation pixels inside contour)
        mask_temp = np.zeros((w_rows, w_cols), dtype=np.uint8)
        cv2.drawContours(mask_temp, [cnt], -1, 255, -1)
        colorful_pixels = cv2.countNonZero(cv2.bitwise_and(sat_mask_50, mask_temp))
        color_ratio = colorful_pixels / area if area > 0 else 0
        
        # Color score: boost for colorful objects, but don't penalize monochrome ones
        if color_ratio > 0.5:
            color_score = 1.0  # Very colorful
        elif color_ratio > 0.2:
            color_score = 0.8  # Somewhat colorful
        else:
            color_score = 0.5  # Monochrome - neutral score
        
        # 7. Method bonus (edge detection with large morphology is best for multi-part objects)
        method_bonus = {
            'edge_large': 1.15,  # Best for objects like Rubik's cube
            'edge_small': 1.0,
            'color': 1.1,       # Good for colorful objects
            'adaptive': 0.9,
            'otsu': 0.85
        }.get(method_name, 1.0)
        
        # Combined score
        total_score = (
            center_score * 0.15 +
            aspect_score * 0.10 +
            size_score * 0.20 +
            solidity_score * 0.25 +
            color_score * 0.30
        ) * edge_penalty * method_bonus
        
        if total_score > best_score:
            best_score = total_score
            best_contour = cnt
    
    if best_contour is None:
        return {'success': False, 'error': 'Could not find suitable object boundary'}
    
    # ========== STEP 5: Create final boundary ==========
    # Scale contour back to original image size
    if scale != 1.0:
        best_contour = (best_contour / scale).astype(np.int32)
    
    # Get convex hull for clean boundary
    hull = cv2.convexHull(best_contour)
    
    # Simplify to polygon
    perimeter = cv2.arcLength(hull, True)
    epsilon = 0.015 * perimeter
    approx = cv2.approxPolyDP(hull, epsilon, True)
        
    # Ensure proper shape
    if len(approx.shape) == 2:
        approx = approx.reshape(-1, 1, 2)
    
    # Calculate metrics
    final_area = cv2.contourArea(approx)
    final_perimeter = cv2.arcLength(approx, True)
    
    # ========== CREATE OUTPUT IMAGES ==========
    # Create mask from the detected contour
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.fillPoly(mask, [approx], 255)
    
    # Boundary image (green outline on original)
    boundary_img = image.copy()
    cv2.drawContours(boundary_img, [approx], -1, (0, 255, 0), 4, cv2.LINE_AA)
        
    # Overlay image (semi-transparent green fill + outline)
    overlay = image.copy()
    green_fill = np.zeros_like(image)
    green_fill[mask > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(image, 0.7, green_fill, 0.3, 0)
    cv2.drawContours(overlay, [approx], -1, (0, 255, 0), 4, cv2.LINE_AA)
    
    # Get boundary points
    boundary_points = approx.reshape(-1, 2).tolist()
    
    # Encode images
    original_encoded = encode_image_to_base64(image)
    boundary_encoded = encode_image_to_base64(boundary_img)
    overlay_encoded = encode_image_to_base64(overlay)
    
    if not all([original_encoded, boundary_encoded, overlay_encoded]):
        return {'success': False, 'error': 'Failed to encode result images'}
    
    return {
        'success': True,
        'original_image': original_encoded,
        'boundary': boundary_encoded,
        'boundary_overlay': overlay_encoded,
        'boundary_points': boundary_points,
        'area': float(final_area),
        'perimeter': float(final_perimeter)
    }

def points_to_contour(pts):
    """
    Order points around centroid by angle for a non-self-crossing polygon
    Matches Assignment3 pointsToContourMat logic
    """
    if len(pts) < 3:
        return None
    
    pts = np.array(pts, dtype=np.float32)
    # Calculate centroid
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    
    # Calculate angles and sort
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    sorted_indices = np.argsort(angles)
    ordered_points = pts[sorted_indices].astype(np.int32)
    
    return ordered_points.reshape(-1, 1, 2)

def segment_with_aruco_handler(image_data, use_corners=True):
    """
    Problem 4: Object Segmentation using ArUco Markers
    Matches Assignment3 segAruco logic exactly:
    - Collects all 4 corners of each marker (or centers if use_corners=False)
    - Orders points around centroid by angle (not convex hull)
    - Creates contour from ordered points
    """
    image = decode_base64_image(image_data)
    if image is None:
        return {'success': False, 'error': 'Invalid image'}
    
    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Try multiple ArUco dictionaries to maximize detection
    aruco_dicts = [
        cv2.aruco.DICT_6X6_250,      # Most common, matches Assignment3
        cv2.aruco.DICT_4X4_250,      # Smaller markers
        cv2.aruco.DICT_5X5_250,      # Medium markers
        cv2.aruco.DICT_7X7_250,      # Larger markers
        cv2.aruco.DICT_ARUCO_ORIGINAL,  # Original ArUco
        cv2.aruco.DICT_APRILTAG_36h11,  # AprilTag format
    ]
    
    corners = None
    ids = None
    detected_dict_name = None
    
    # Configure detector parameters for better detection
    aruco_params = cv2.aruco.DetectorParameters()
    # Relax thresholds for better detection
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10
    aruco_params.adaptiveThreshConstant = 7
    aruco_params.minMarkerPerimeterRate = 0.01  # Allow smaller markers
    aruco_params.maxMarkerPerimeterRate = 4.0   # Allow larger markers
    aruco_params.polygonalApproxAccuracyRate = 0.05
    aruco_params.minCornerDistanceRate = 0.02
    aruco_params.minDistanceToBorder = 1
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    # Try each dictionary
    for dict_type in aruco_dicts:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    
    # Try each dictionary
    for dict_type in aruco_dicts:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        
        # Detect markers using compatible API
        if hasattr(cv2.aruco, 'ArucoDetector'):
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            corners_temp, ids_temp, rejected = detector.detectMarkers(gray)
        else:
            corners_temp, ids_temp, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        
        if ids_temp is not None and len(ids_temp) > 0:
            corners = corners_temp
            ids = ids_temp
            detected_dict_name = str(dict_type)
            break
    
    # If still no detection, try with enhanced preprocessing
    if ids is None or len(ids) == 0:
        preprocessing_methods = []
        
        # Method 1: CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        preprocessing_methods.append(('CLAHE', enhanced))
        
        # Method 2: Gaussian blur + threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessing_methods.append(('Otsu', binary1))
        
        # Method 3: Adaptive thresholding
        binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        preprocessing_methods.append(('Adaptive', binary2))
        
        # Method 4: Inverted adaptive thresholding
        binary3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
        binary3 = cv2.bitwise_not(binary3)
        preprocessing_methods.append(('AdaptiveInv', binary3))
        
        # Method 5: Sharpen then threshold
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
        _, binary4 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessing_methods.append(('Sharpened', binary4))
        
        # Method 6: Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(binary1, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        preprocessing_methods.append(('Morphed', morph))
        
        for method_name, processed_img in preprocessing_methods:
            for dict_type in aruco_dicts:
                aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
                
                if hasattr(cv2.aruco, 'ArucoDetector'):
                    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
                    corners_temp, ids_temp, _ = detector.detectMarkers(processed_img)
                else:
                    corners_temp, ids_temp, _ = cv2.aruco.detectMarkers(processed_img, aruco_dict, parameters=aruco_params)
                
                if ids_temp is not None and len(ids_temp) > 0:
                    corners = corners_temp
                    ids = ids_temp
                    break
            
            if ids is not None and len(ids) > 0:
                break
    
    # If still no detection, try resizing the image (markers might be too small or too large)
    if ids is None or len(ids) == 0:
        for scale_factor in [0.5, 2.0, 1.5, 0.75]:
            resized = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            
            for dict_type in aruco_dicts[:4]:  # Try first 4 dictionaries only for speed
                aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
                
                if hasattr(cv2.aruco, 'ArucoDetector'):
                    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
                    corners_temp, ids_temp, _ = detector.detectMarkers(resized)
                else:
                    corners_temp, ids_temp, _ = cv2.aruco.detectMarkers(resized, aruco_dict, parameters=aruco_params)
                
                if ids_temp is not None and len(ids_temp) > 0:
                    # Scale corners back to original size
                    corners = [c / scale_factor for c in corners_temp]
                    ids = ids_temp
                    break
            
            if ids is not None and len(ids) > 0:
                break
    
    # Last resort: try with rotations (in case marker is tilted)
    if ids is None or len(ids) == 0:
        for angle in [90, 180, 270, 45, -45]:
            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new image bounds
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            rotated = cv2.warpAffine(gray, M, (new_w, new_h), borderValue=255)
            
            for dict_type in aruco_dicts[:3]:
                aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
                
                if hasattr(cv2.aruco, 'ArucoDetector'):
                    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
                    corners_temp, ids_temp, _ = detector.detectMarkers(rotated)
                else:
                    corners_temp, ids_temp, _ = cv2.aruco.detectMarkers(rotated, aruco_dict, parameters=aruco_params)
                
                if ids_temp is not None and len(ids_temp) > 0:
                    # Transform corners back to original orientation
                    M_inv = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), -angle, 1.0)
                    M_inv[0, 2] += center[0] - (new_w / 2)
                    M_inv[1, 2] += center[1] - (new_h / 2)
                    
                    transformed_corners = []
                    for corner_set in corners_temp:
                        pts = corner_set[0]
                        pts_homogeneous = np.hstack([pts, np.ones((4, 1))])
                        pts_transformed = pts_homogeneous @ M_inv.T
                        transformed_corners.append(pts_transformed.reshape(1, 4, 2))
                    
                    corners = transformed_corners
                    ids = ids_temp
                    break
            
            if ids is not None and len(ids) > 0:
                break
    
    if ids is None or len(ids) == 0:
        return {
            'success': False,
            'error': 'No ArUco markers detected. Tried multiple dictionary types, preprocessing methods, and rotations. Please ensure markers are clearly visible, properly printed, and have sufficient white border around them.'
        }
    
    # Draw detected markers
    result = image.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(result, corners, ids)
    
    # Extract marker corner points (matching Assignment3 cornersToPointList logic)
    marker_points = []
    for i, corner_set in enumerate(corners):
        corner_set = corner_set[0]  # Get first (and only) marker (shape: 4x2)
        if use_corners:
            # Use all 4 corners of each marker (matching Assignment3 useCorners=true)
            for point in corner_set:
                marker_points.append([int(point[0]), int(point[1])])
        else:
            # Use center of marker
            cx = np.mean(corner_set[:, 0])
            cy = np.mean(corner_set[:, 1])
            marker_points.append([int(cx), int(cy)])
    
    if len(marker_points) < 3:
        return {
            'success': False,
            'error': f'Only {len(marker_points)} marker points found. Need at least 3 points for boundary detection.'
        }
    
    # Order points around centroid by angle (matching Assignment3 pointsToContourMat logic)
    ordered_contour = points_to_contour(marker_points)
    
    if ordered_contour is None:
        return {
            'success': False,
            'error': 'Failed to create ordered contour from marker points.'
        }
    
    # Create mask from ordered contour (not convex hull - matches Assignment3)
        mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [ordered_contour], 255)
    
    # Draw boundary using ordered contour
    cv2.drawContours(result, [ordered_contour], -1, (0, 255, 0), 3)
    
    # Apply mask to original image
    segmented = image.copy()
    segmented[mask == 0] = [0, 0, 0]  # Black background
        
    # Create overlay (matching Assignment3 blending: 0.65 original + 0.35 green fill)
    fill_rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    fill_rgb[mask > 0] = [0, 255, 0]  # Green fill (BGR)
    result_overlay = cv2.addWeighted(image, 0.65, fill_rgb, 0.35, 0)
    cv2.drawContours(result_overlay, [ordered_contour], -1, (0, 255, 0), 2, cv2.LINE_AA)
        
    # Get boundary points (from ordered contour, not hull)
    boundary_points = ordered_contour.reshape(-1, 2).tolist()
    
    # Calculate area and perimeter from ordered contour
    area = float(cv2.contourArea(ordered_contour))
    perimeter = float(cv2.arcLength(ordered_contour, True))
    
    # Encode images
    original_encoded = encode_image_to_base64(image)
    markers_encoded = encode_image_to_base64(result)
    segmented_encoded = encode_image_to_base64(segmented)
    overlay_encoded = encode_image_to_base64(result_overlay)
    
    if not all([original_encoded, markers_encoded, segmented_encoded, overlay_encoded]):
        return {'success': False, 'error': 'Failed to encode result images'}
        
        return {
            'success': True,
        'original_image': original_encoded,
        'markers_detected': markers_encoded,
        'segmented': segmented_encoded,
        'segmented_overlay': overlay_encoded,
            'marker_count': int(len(ids)),
            'boundary_points': boundary_points,
        'area': area,
        'perimeter': perimeter
    }

def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two masks
    Matches Assignment3 calculateIoU logic
    """
    if mask1 is None or mask2 is None:
        return 0.0
    if mask1.shape != mask2.shape:
        return 0.0
    
    # Convert to binary if needed
    if len(mask1.shape) == 3:
        mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
    if len(mask2.shape) == 3:
        mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
    
    # Threshold to binary
    _, mask1_bin = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
    _, mask2_bin = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate intersection and union
    intersection = cv2.bitwise_and(mask1_bin, mask2_bin)
    union = cv2.bitwise_or(mask1_bin, mask2_bin)
    
    inter_area = cv2.countNonZero(intersection)
    union_area = cv2.countNonZero(union)
    
    return inter_area / union_area if union_area > 0 else 0.0

def compare_with_sam2_handler(image_data, sam2_result_data):
    """
    Problem 5: Compare ArUco segmentation with SAM2 results
    Matches Assignment3 compareSegmentation logic:
    - Gets ArUco mask
    - Gets SAM2 mask (from uploaded image)
    - Calculates IoU and metrics
    - Creates side-by-side comparison visualization
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
    
    # Resize SAM2 result to match original if needed
    if sam2_result.shape[:2] != image.shape[:2]:
        sam2_result = cv2.resize(sam2_result, (image.shape[1], image.shape[0]))
    
    # Get ArUco mask (reuse segmentation logic)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    aruco_mask = None
    if ids is not None and len(ids) > 0:
        # Extract marker points (use all corners)
        marker_points = []
        for i, corner_set in enumerate(corners):
            corner_set = corner_set[0]
            for point in corner_set:
                marker_points.append([int(point[0]), int(point[1])])
        
        if len(marker_points) >= 3:
            ordered_contour = points_to_contour(marker_points)
            if ordered_contour is not None:
                aruco_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(aruco_mask, [ordered_contour], 255)
    
    # Convert SAM2 result to mask (assume it's a segmentation mask or image)
    sam2_gray = cv2.cvtColor(sam2_result, cv2.COLOR_BGR2GRAY) if len(sam2_result.shape) == 3 else sam2_result
    _, sam2_mask = cv2.threshold(sam2_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate metrics (matching Assignment3 calculateMetrics)
    if aruco_mask is not None:
        iou = calculate_iou(aruco_mask, sam2_mask)
        aruco_area = cv2.countNonZero(aruco_mask)
        sam2_area = cv2.countNonZero(sam2_mask)
        
        # Calculate difference
        diff_mask = cv2.absdiff(aruco_mask, sam2_mask)
        diff_area = cv2.countNonZero(diff_mask)
    else:
        iou = 0.0
        aruco_area = 0
        sam2_area = cv2.countNonZero(sam2_mask)
        diff_area = sam2_area
    
    # Create comparison visualization (matching Assignment3: 3-panel layout)
    rows, cols = image.shape[:2]
    comparison = np.zeros((rows, cols * 3, 3), dtype=np.uint8)
    
    # Left: ArUco result
    aruco_overlay = image.copy()
    if aruco_mask is not None:
        aruco_fill = np.zeros((rows, cols, 3), dtype=np.uint8)
        aruco_fill[aruco_mask > 0] = [0, 255, 0]  # Green (BGR)
        aruco_overlay = cv2.addWeighted(image, 0.65, aruco_fill, 0.35, 0)
    comparison[:, :cols] = aruco_overlay
    
    # Middle: SAM2 result
    sam2_overlay = image.copy()
    sam2_fill = np.zeros((rows, cols, 3), dtype=np.uint8)
    sam2_fill[sam2_mask > 0] = [0, 0, 255]  # Red (BGR)
    sam2_overlay = cv2.addWeighted(image, 0.65, sam2_fill, 0.35, 0)
    comparison[:, cols:cols*2] = sam2_overlay
    
    # Right: Difference
    diff_overlay = image.copy()
    diff_fill = np.zeros((rows, cols, 3), dtype=np.uint8)
    diff_fill[diff_mask > 0] = [0, 255, 255]  # Yellow (BGR)
    diff_overlay = cv2.addWeighted(image, 0.7, diff_fill, 0.3, 0)
    comparison[:, cols*2:] = diff_overlay
    
    # Add labels
    cv2.putText(comparison, 'ArUco', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, 'SAM2', (cols + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(comparison, 'Diff', (cols * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Encode images
    original_encoded = encode_image_to_base64(image)
    sam2_encoded = encode_image_to_base64(sam2_result)
    comparison_encoded = encode_image_to_base64(comparison)
    
    if not all([original_encoded, sam2_encoded, comparison_encoded]):
        return {'success': False, 'error': 'Failed to encode result images'}
    
    return {
        'success': True,
        'original_image': original_encoded,
        'sam2_result': sam2_encoded,
        'comparison': comparison_encoded,
        'metrics': {
            'iou': float(iou),
            'aruco_area': int(aruco_area),
            'sam2_area': int(sam2_area),
            'diff_area': int(diff_area)
        },
        'note': f'IoU: {(iou * 100):.1f}% | ArUco: {aruco_area}px | SAM2: {sam2_area}px | Diff: {diff_area}px'
    }
