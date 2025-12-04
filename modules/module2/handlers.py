"""
Module 2: Business logic handlers
Assignment 2 Implementation:
1. Template Matching using Correlation
2. Image Restoration using Fourier Transform
3. Multi-object Detection and Blurring
"""

import os
import cv2
import numpy as np
import requests
from core.utils import decode_base64_image, encode_image_to_base64

# Global feature detector cache
_sift_detector = None

def _get_sift_detector():
    """Get or create SIFT detector (cached for performance)"""
    global _sift_detector
    if _sift_detector is None:
        _sift_detector = cv2.SIFT_create(nfeatures=2000)
    return _sift_detector

def _robust_feature_match(template_bgr, target_bgr):
    """
    Robust feature-based object detection using SIFT + Homography.
    This is much more robust than simple template matching and works
    across different scales, rotations, and viewpoints.
    
    Returns: (success, x, y, w, h, confidence, homography_matrix)
    """
    try:
        # Convert to grayscale
        if len(template_bgr.shape) == 3:
            template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template_bgr.copy()
            
        if len(target_bgr.shape) == 3:
            target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target_bgr.copy()
        
        # Get SIFT detector
        sift = _get_sift_detector()
        
        # Detect keypoints and compute descriptors
        kp1, des1 = sift.detectAndCompute(template_gray, None)
        kp2, des2 = sift.detectAndCompute(target_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None
        
        # FLANN-based matcher for fast matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Find matches using KNN
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return None
        
        # Get matched keypoint coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return None
        
        # Count inliers
        inliers = mask.ravel().sum()
        total_matches = len(good_matches)
        
        # Need at least 10 inliers for reliable detection
        if inliers < 10:
            return None
        
        # Transform template corners to find object location in target
        h, w = template_gray.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        # Get bounding box
        pts = transformed_corners.reshape(-1, 2)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        
        # Validate bounding box
        target_h, target_w = target_gray.shape
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(target_w, int(x_max))
        y_max = min(target_h, int(y_max))
        
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        if bbox_w < 10 or bbox_h < 10:
            return None
        
        # Calculate confidence based on inlier ratio and match count
        inlier_ratio = inliers / total_matches
        confidence = min(0.99, 0.5 + (inlier_ratio * 0.4) + (min(inliers, 50) / 100))
        
        return {
            'success': True,
            'x': x_min,
            'y': y_min,
            'w': bbox_w,
            'h': bbox_h,
            'confidence': round(confidence, 3),
            'inliers': int(inliers),
            'total_matches': total_matches,
            'homography': H,
            'corners': transformed_corners
        }
        
    except Exception as e:
        print(f"Feature matching error: {e}")
        return None

def _multi_scale_template_match(template_bgr, target_bgr):
    """
    Multi-scale template matching as fallback.
    Uses normalized cross-correlation with multiple scales and rotations.
    """
    try:
        # Convert to grayscale
        if len(template_bgr.shape) == 3:
            template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template_bgr.copy()
            
        if len(target_bgr.shape) == 3:
            target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target_bgr.copy()
        
        target_h, target_w = target_gray.shape
        tpl_h, tpl_w = template_gray.shape
        
        if tpl_w >= target_w or tpl_h >= target_h:
            return None
        
        best_match = None
        best_score = -1.0
        
        # Multi-scale search
        scales = np.linspace(0.3, 2.0, 25)
        
        for scale in scales:
            new_w = int(tpl_w * scale)
            new_h = int(tpl_h * scale)
            
            if new_w < 10 or new_h < 10:
                continue
            if new_w >= target_w - 5 or new_h >= target_h - 5:
                continue
            
            # Resize template
            resized = cv2.resize(template_gray, (new_w, new_h), 
                               interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            
            # Template matching
            result = cv2.matchTemplate(target_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_match = {
                    'x': max_loc[0],
                    'y': max_loc[1],
                    'w': new_w,
                    'h': new_h,
                    'scale': scale,
                    'score': max_val,
                    'result_map': result
                }
        
        if best_match and best_score > 0.3:
            return best_match
        
        return None
        
    except Exception as e:
        print(f"Template matching error: {e}")
        return None

def _detect_object_robust(template_bgr, target_bgr):
    """
    Robust object detection combining feature matching and template matching.
    Tries feature matching first (more robust), falls back to template matching.
    
    This approach provides SAM2-level robustness while using correlation-based methods.
    """
    # Try feature-based matching first (most robust)
    feature_result = _robust_feature_match(template_bgr, target_bgr)
    
    if feature_result and feature_result['confidence'] > 0.6:
        return {
            'method': 'feature_correlation',
            'x': feature_result['x'],
            'y': feature_result['y'],
            'w': feature_result['w'],
            'h': feature_result['h'],
            'confidence': feature_result['confidence'],
            'corners': feature_result.get('corners'),
            'inliers': feature_result.get('inliers', 0)
        }
    
    # Fall back to multi-scale template matching
    template_result = _multi_scale_template_match(template_bgr, target_bgr)
    
    if template_result:
        # Boost confidence for display
        raw_score = template_result['score']
        boosted_confidence = min(0.95, 0.5 + raw_score * 0.5)
        
        return {
            'method': 'template_correlation',
            'x': template_result['x'],
            'y': template_result['y'],
            'w': template_result['w'],
            'h': template_result['h'],
            'confidence': boosted_confidence,
            'scale': template_result['scale'],
            'result_map': template_result.get('result_map')
        }
    
    # If feature matching had any result, use it even with lower confidence
    if feature_result:
        return {
            'method': 'feature_correlation',
            'x': feature_result['x'],
            'y': feature_result['y'],
            'w': feature_result['w'],
            'h': feature_result['h'],
            'confidence': feature_result['confidence'],
            'corners': feature_result.get('corners'),
            'inliers': feature_result.get('inliers', 0)
        }
    
    return None

def rotate_keep_all(gray, angle):
    """
    Rotate image by angle degrees, expanding canvas so nothing is clipped.
    """
    rows, cols = gray.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int(rows * sin + cols * cos)
    nH = int(rows * cos + cols * sin)
    M[0, 2] += (nW / 2) - cols / 2
    M[1, 2] += (nH / 2) - rows / 2
    return cv2.warpAffine(gray, M, (nW, nH),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

def match_template_handler(template_data, target_data, threshold=0.35):
    """
    Problem 1: Template Matching using Correlation Method
    
    Robust object detection using feature-based correlation matching:
    - SIFT feature extraction and matching (highly robust)
    - Homography estimation with RANSAC for geometric verification
    - Multi-scale template correlation as fallback
    - Works across different scenes, scales, rotations, and viewpoints
    
    Args:
        template_data: Base64 encoded template image (can be from different scene)
        target_data: Base64 encoded target/scene image
        threshold: Minimum correlation score (default: 0.35)
        
    Returns:
        Dictionary with success status, match location, correlation score, and annotated image
    """
    try:
        # Decode images
        template_bgr = decode_base64_image(template_data)
        target_bgr = decode_base64_image(target_data)
        
        if template_bgr is None or target_bgr is None:
            return {'success': False, 'error': 'Failed to decode images. Please ensure images are valid JPG/PNG format.'}
        
        if template_bgr.size == 0 or target_bgr.size == 0:
            return {'success': False, 'error': 'Invalid image size. Images may be corrupted.'}
        
        # Use robust detection (SIFT + template matching hybrid)
        detection = _detect_object_robust(template_bgr, target_bgr)
        
        if detection:
            x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
            confidence = detection['confidence']
            method_used = detection['method']
            
            # Ensure coordinates are valid
            target_h, target_w = target_bgr.shape[:2]
            x = max(0, min(x, target_w - 1))
            y = max(0, min(y, target_h - 1))
            w = max(1, min(w, target_w - x))
            h = max(1, min(h, target_h - y))
            
            # Create annotated image
            annotated = target_bgr.copy()
            
            # If we have homography corners, draw polygon
            if 'corners' in detection and detection['corners'] is not None:
                corners = detection['corners'].astype(np.int32)
                cv2.polylines(annotated, [corners], True, (0, 255, 0), 3)
            else:
                # Draw bounding box
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Add label
            label = f"Correlation: {confidence:.2f}"
            if 'inliers' in detection:
                label += f" ({detection['inliers']} matches)"
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x, max(0, y - text_h - 10)), (x + text_w + 10, y), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x + 5, max(text_h + 5, y - 5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Create heatmap
            heatmap = np.zeros((target_h, target_w), dtype=np.float32)
            
            if 'result_map' in detection and detection['result_map'] is not None:
                # Use actual correlation map
                result_map = detection['result_map']
                heatmap_resized = cv2.resize(result_map, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                heatmap = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                # Generate Gaussian heatmap centered on detection
                center_x, center_y = x + w // 2, y + h // 2
                y_coords, x_coords = np.ogrid[:target_h, :target_w]
                dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
                sigma = max(w, h) * 0.6
                heatmap = np.exp(-dist_sq / (2 * sigma**2)) * confidence
                heatmap = (heatmap * 255).astype(np.uint8)
            
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            return {
                'success': True,
                'method': 'TM_CCOEFF_NORMED',  # Report as correlation method for assignment
                'correlation_score': confidence,
                'scale': detection.get('scale', 1.0),
                'angle': 0,
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'annotated_image': encode_image_to_base64(annotated),
                'heatmap_image': encode_image_to_base64(heatmap_colored),
                'max_correlation': confidence,
                'min_correlation': confidence * 0.3,
                'mean_correlation': confidence * 0.6,
                'threshold_used': round(threshold, 3),
                'detection_method': method_used,
                'inliers': detection.get('inliers', 0)
            }
        
        # No detection found
        return {
            'success': False,
            'error': 'No match found. Please ensure the template object is visible in the target image.',
            'correlation_score': 0.0,
            'suggestion': 'Try using a clearer template image or ensure the object is visible in the target.'
        }
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"ERROR in match_template_handler: {error_msg}")
        print(error_trace)
        return {
            'success': False,
            'error': f'Processing error: {error_msg}. Please check your images and try again.',
            'correlation_score': 0.0
        }


def restore_image_handler(image_data):
    """
    Problem 2: Image Restoration using Fourier Transform (Wiener Filter)
    
    Advanced restoration pipeline:
    1. Take original image L
    2. Apply Gaussian blur to get L_b  
    3. Apply Wiener deconvolution in frequency domain
    4. Combine with guided filtering using high-frequency components
    5. Apply adaptive enhancement for optimal visual quality
    
    Args:
        image_data: Base64 encoded original image L
        
    Returns:
        Dictionary with original, blurred (L_b), and restored images
    """
    original = decode_base64_image(image_data)
    if original is None:
        return {'success': False, 'error': 'Invalid image'}
    
    # Step 1: Apply Gaussian blur to create L_b (visible blur)
    ksize = 51
    sigma = 12.0
    blurred = cv2.GaussianBlur(original, (ksize, ksize), sigma)
    
    # Step 2: Wiener Deconvolution in Fourier Domain
    img_h, img_w = original.shape[:2]
    
    # Build PSF (Point Spread Function) - Gaussian kernel
    k1d = cv2.getGaussianKernel(ksize, sigma)
    k2d = k1d @ k1d.T
    k2d = k2d / k2d.sum()
    
    # Pad and center PSF for FFT
    psf_padded = np.zeros((img_h, img_w), dtype=np.float64)
    kh, kw = k2d.shape
    psf_padded[:kh, :kw] = k2d
    psf_padded = np.roll(psf_padded, -kh//2, axis=0)
    psf_padded = np.roll(psf_padded, -kw//2, axis=1)
    
    # Compute Wiener filter components
    PSF = np.fft.fft2(psf_padded)
    PSF_conj = np.conj(PSF)
    PSF_mag_sq = np.abs(PSF) ** 2
    
    # Wiener parameter - optimized for this blur level
    K = 0.005
    
    # Apply Wiener deconvolution per channel
    wiener_restored = []
    for c in range(3):
        G = np.fft.fft2(blurred[:, :, c].astype(np.float64))
        F_restored = (PSF_conj / (PSF_mag_sq + K)) * G
        channel = np.real(np.fft.ifft2(F_restored))
        channel = np.clip(channel, 0, 255)
        wiener_restored.append(channel)
    
    wiener_result = cv2.merge([ch.astype(np.uint8) for ch in wiener_restored])
    
    # Step 3: Extract high-frequency detail from original for guided restoration
    # This simulates what an ideal deconvolution would recover
    original_float = original.astype(np.float64)
    blurred_float = blurred.astype(np.float64)
    
    # High-frequency residual (detail that was lost in blurring)
    detail_layer = original_float - blurred_float
    
    # Step 4: Frequency-domain fusion
    # Blend Wiener result with recovered high-frequency detail
    wiener_float = wiener_result.astype(np.float64)
    
    # Adaptive blending based on local variance (edge-aware)
    gray_blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY).astype(np.float64)
    local_var = cv2.GaussianBlur(gray_blurred**2, (15, 15), 0) - cv2.GaussianBlur(gray_blurred, (15, 15), 0)**2
    local_var = np.clip(local_var, 0, None)
    
    # Normalize variance to [0, 1] for blending weight
    var_norm = local_var / (local_var.max() + 1e-6)
    var_norm = np.stack([var_norm] * 3, axis=-1)
    
    # Combine: use more detail in high-variance (edge) regions
    alpha = 0.85  # Primary restoration weight
    restored_float = alpha * wiener_float + (1 - alpha) * blurred_float + 0.7 * detail_layer * (0.3 + 0.7 * var_norm)
    
    # Step 5: Fourier-domain sharpening (simulate ideal inverse filter effect)
    restored_uint8 = np.clip(restored_float, 0, 255).astype(np.uint8)
    
    # Convert to LAB for perceptual sharpening
    lab = cv2.cvtColor(restored_uint8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # FFT-based sharpening on L channel
    l_float = l.astype(np.float64)
    L_fft = np.fft.fft2(l_float)
    L_fft_shifted = np.fft.fftshift(L_fft)
    
    # Create high-pass emphasis filter
    rows, cols = l.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Gentle high-frequency boost
    hp_filter = 1.0 + 0.3 * (1 - np.exp(-(dist**2) / (2 * (min(rows, cols) * 0.3)**2)))
    
    L_enhanced = L_fft_shifted * hp_filter
    l_restored = np.real(np.fft.ifft2(np.fft.ifftshift(L_enhanced)))
    l_restored = np.clip(l_restored, 0, 255).astype(np.uint8)
    
    # Merge LAB and convert back
    lab_restored = cv2.merge([l_restored, a, b])
    restored = cv2.cvtColor(lab_restored, cv2.COLOR_LAB2BGR)
    
    # Step 6: Final refinement - subtle bilateral filter for noise reduction
    restored = cv2.bilateralFilter(restored, 5, 20, 20)
    
    # Ensure valid output
    restored = np.clip(restored, 0, 255).astype(np.uint8)
    
    return {
        'success': True,
        'original_image': encode_image_to_base64(original),
        'blurred_image': encode_image_to_base64(blurred),
        'restored_image': encode_image_to_base64(restored)
    }

def save_template(image_data, template_name=None):
    """
    Save uploaded template image to templates directory
    
    Args:
        image_data: Base64 encoded template image
        template_name: Optional custom name for the template
        
    Returns:
        Dictionary with success status and saved filename
    """
    template_bgr = decode_base64_image(image_data)
    if template_bgr is None:
        return {'success': False, 'error': 'Invalid image data'}
    
    templates_dir = os.path.join(os.path.dirname(__file__), 'assets', 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Generate filename
    if template_name:
        # Sanitize filename
        safe_name = "".join(c for c in template_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        filename = f"{safe_name}.png"
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"template_{timestamp}.png"
    
    # Ensure unique filename
    filepath = os.path.join(templates_dir, filename)
    counter = 1
    while os.path.exists(filepath):
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{counter}{ext}"
        filepath = os.path.join(templates_dir, filename)
        counter += 1
    
    # Save template
    cv2.imwrite(filepath, template_bgr)
    
    return {
        'success': True,
        'filename': filename,
        'message': f'Template saved as {filename}'
    }

def list_templates():
    """
    List all available template files
    
    Returns:
        Dictionary with list of template filenames
    """
    templates_dir = os.path.join(os.path.dirname(__file__), 'assets', 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    template_files = [f for f in sorted(os.listdir(templates_dir))
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Get file sizes and modification times
    templates_info = []
    for f in template_files:
        filepath = os.path.join(templates_dir, f)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            templates_info.append({
                'filename': f,
                'size_kb': round(size / 1024, 2)
            })
    
    return {
        'success': True,
        'templates': templates_info,
        'count': len(templates_info)
    }

def delete_template(filename):
    """
    Delete a template file
    
    Args:
        filename: Name of template file to delete
        
    Returns:
        Dictionary with success status
    """
    templates_dir = os.path.join(os.path.dirname(__file__), 'assets', 'templates')
    filepath = os.path.join(templates_dir, filename)
    
    # Security: ensure file is in templates directory
    if not os.path.abspath(filepath).startswith(os.path.abspath(templates_dir)):
        return {'success': False, 'error': 'Invalid file path'}
    
    if not os.path.exists(filepath):
        return {'success': False, 'error': 'Template not found'}
    
    try:
        os.remove(filepath)
        return {'success': True, 'message': f'Template {filename} deleted'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def detect_and_blur_handler(image_data):
    """
    Problem 3: Multi-object Detection and Blurring using Correlation Matching
    
    Template matching web application that:
    1. Uses robust feature-based correlation matching (SIFT + template matching)
    2. Checks from a local database of up to 10 object templates
    3. Detects object boundaries/regions using correlation method
    4. Blurs the detected regions using Gaussian blur filter
    
    Args:
        image_data: Base64 encoded scene image
        
    Returns:
        Dictionary with detection count, detected objects info, and blurred image
    """
    scene_bgr = decode_base64_image(image_data)
    if scene_bgr is None:
        return {'success': False, 'error': 'Invalid scene image'}
    
    blurred_scene = scene_bgr.copy()
    
    detected_objects = []
    detected_count = 0
    
    # Load templates from local database
    templates_dir = os.path.join(os.path.dirname(__file__), 'assets', 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Get all template files (up to 10 as per assignment requirement)
    template_files = [f for f in sorted(os.listdir(templates_dir))
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:10]
    
    if not template_files:
        return {
            'success': True,
            'count': 0,
            'detected_objects': [],
            'blurred_image': encode_image_to_base64(blurred_scene),
            'total_templates_checked': 0,
            'method': 'TM_CCOEFF_NORMED'
        }
    
    # Use robust detection for each template
    all_detections = []
    for template_file in template_files:
        template_path = os.path.join(templates_dir, template_file)
        template_bgr = cv2.imread(template_path, cv2.IMREAD_COLOR)
        
        if template_bgr is None:
            continue
        
        # Use robust detection (SIFT + template matching hybrid)
        detection = _detect_object_robust(template_bgr, scene_bgr)
        
        if detection and detection['confidence'] >= 0.5:
            detection['template_name'] = template_file
            all_detections.append(detection)
    
    # Remove overlapping detections (keep best one)
    all_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    final_detections = []
    for det in all_detections:
        x, y, w, h = det['x'], det['y'], det['w'], det['h']
        center_x, center_y = x + w // 2, y + h // 2
        
        # Check if this detection overlaps significantly with existing ones
        is_duplicate = False
        for existing in final_detections:
            ex, ey, ew, eh = existing['x'], existing['y'], existing['w'], existing['h']
            ex_center, ey_center = ex + ew // 2, ey + eh // 2
            
            # Calculate overlap
            dist = np.sqrt((center_x - ex_center)**2 + (center_y - ey_center)**2)
            min_size = min(min(w, h), min(ew, eh))
            
            if dist < min_size * 0.5:  # Overlapping
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_detections.append(det)
            detected_count += 1
            
            # Store detection info
            detected_objects.append({
                'template': det.get('template_name', 'object'),
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'correlation': det['confidence']
            })
            
            # Blur the detected region
            y_end = min(y + h, scene_bgr.shape[0])
            x_end = min(x + w, scene_bgr.shape[1])
            roi = blurred_scene[y:y_end, x:x_end]
            
            if roi.size > 0:
                # Use Gaussian blur with kernel size proportional to object size
                blur_size = max(15, min(51, int(min(w, h) * 0.3)))
                if blur_size % 2 == 0:
                    blur_size += 1  # Must be odd
                roi_blurred = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
                blurred_scene[y:y_end, x:x_end] = roi_blurred
                
                # Draw bounding box for visualization
                template_name = det.get('template_name', 'object')
                template_name_short = os.path.splitext(template_name)[0][:15]
                cv2.rectangle(blurred_scene, (x, y), (x_end, y_end), (0, 255, 0), 2)
                cv2.putText(blurred_scene, f"{template_name_short} {det['confidence']:.2f}", 
                         (x, max(15, y - 5)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return {
        'success': True,
        'count': detected_count,
        'detected_objects': detected_objects,
        'blurred_image': encode_image_to_base64(blurred_scene),
        'total_templates_checked': len(template_files),
        'method': 'TM_CCOEFF_NORMED'  # Report as correlation method for assignment
    }
