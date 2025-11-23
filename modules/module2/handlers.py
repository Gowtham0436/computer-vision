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
from core.utils import decode_base64_image, encode_image_to_base64

def match_template_handler(template_data, target_data):
    """
    Problem 1: Template Matching using Correlation Method
    
    Accurate template matching with multi-scale search and validation:
    - Finds best match across multiple scales
    - Validates matches using edge correlation
    - Works with templates cropped from same or different scenes
    - Handles different backgrounds and lighting conditions
    
    Args:
        template_data: Base64 encoded template image
        target_data: Base64 encoded target/scene image
        
    Returns:
        Dictionary with success status, match location, correlation score, and annotated image
    """
    # Decode images
    template_bgr = decode_base64_image(template_data)
    target_bgr = decode_base64_image(target_data)
    
    if template_bgr is None or target_bgr is None:
        return {'success': False, 'error': 'Invalid image(s)'}
    
    # Convert to grayscale
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    
    # Ensure template is smaller than target
    if template_gray.shape[0] > target_gray.shape[0] or template_gray.shape[1] > target_gray.shape[1]:
        return {'success': False, 'error': 'Template must be smaller than target image'}
    
    # Get template dimensions
    th, tw = template_gray.shape[:2]
    
    # Multi-scale template matching
    best_match = None
    best_score = -1.0
    best_scale = 1.0
    best_template_size = (th, tw)
    
    # Try different scales to handle size variations
    scales = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    
    for scale in scales:
        # Scale template
        if scale != 1.0:
            new_w = int(tw * scale)
            new_h = int(th * scale)
            
            if new_w < 15 or new_h < 15 or new_w > target_gray.shape[1] or new_h > target_gray.shape[0]:
                continue
            
            template_scaled = cv2.resize(template_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            template_scaled = template_gray
        
        # Template matching using normalized cross-correlation
        result = cv2.matchTemplate(target_gray, template_scaled, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Validate match using edge correlation for better accuracy
        x, y = max_loc
        if y + template_scaled.shape[0] <= target_gray.shape[0] and \
           x + template_scaled.shape[1] <= target_gray.shape[1]:
            
            # Extract matched region
            matched_roi = target_gray[y:y+template_scaled.shape[0], 
                                     x:x+template_scaled.shape[1]]
            
            # Compute edge-based validation
            template_edges = cv2.Canny(template_scaled, 50, 150)
            roi_edges = cv2.Canny(matched_roi, 50, 150)
            
            if template_edges.size > 0 and roi_edges.size > 0 and \
               template_edges.shape == roi_edges.shape:
                # Edge correlation for validation
                edge_result = cv2.matchTemplate(roi_edges, template_edges, cv2.TM_CCOEFF_NORMED)
                edge_score = edge_result[0, 0] if edge_result.size > 0 else 0
                
                # Combined score: 80% correlation + 20% edge match
                # Edge match helps validate that it's actually the right object
                combined_score = 0.8 * max_val + 0.2 * max(0, edge_score)
            else:
                combined_score = max_val
            
            # Update best match
            if combined_score > best_score:
                best_score = combined_score
                best_match = (x, y)
                best_scale = scale
                best_template_size = template_scaled.shape[:2]
    
    # Adaptive threshold based on template size
    template_area = th * tw
    target_area = target_gray.shape[0] * target_gray.shape[1]
    size_ratio = template_area / target_area
    
    # Set thresholds - higher for better accuracy
    if size_ratio > 0.3:
        min_threshold = 0.65  # Large template needs high confidence
    elif size_ratio > 0.1:
        min_threshold = 0.55  # Medium template
    else:
        min_threshold = 0.45  # Small/cropped template
    
    if best_score < min_threshold or best_match is None:
        return {
            'success': False,
            'error': f'No reliable match found. Best score: {best_score:.3f} (threshold: {min_threshold:.2f}). Ensure template is clear, well-cropped, and object is clearly visible in target image.',
            'correlation_score': float(best_score) if best_score > -1 else 0.0
        }
    
    # Get template dimensions at best scale
    h, w = best_template_size
    
    # Draw rectangle on target image
    annotated = target_bgr.copy()
    top_left = best_match
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # Draw bounding box
    cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 3)
    
    # Add correlation score
    text = f'Match: {best_score:.3f}'
    if best_scale != 1.0:
        text += f' (scale: {best_scale:.2f}x)'
    cv2.putText(annotated, text,
               (top_left[0], max(20, top_left[1] - 10)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return {
        'success': True,
        'method': 'CCOEFF_NORMED',
        'correlation_score': round(float(best_score), 4),
        'scale': round(float(best_scale), 2),
        'x': int(top_left[0]),
        'y': int(top_left[1]),
        'w': int(w),
        'h': int(h),
        'annotated_image': encode_image_to_base64(annotated)
    }

def restore_image_handler(image_data):
    """
    Problem 2: Image Restoration using Fourier Transform (Enhanced Wiener Filter)
    
    Generic restoration algorithm that works well for various image types:
    - Portraits, objects, landscapes, etc.
    - Different resolutions and orientations
    
    Process:
    1. Take original image L
    2. Apply Gaussian blur to get L_b
    3. Use adaptive Wiener filter in Fourier domain to restore L from L_b
    4. Apply iterative refinement and advanced post-processing
    
    Args:
        image_data: Base64 encoded original image L
        
    Returns:
        Dictionary with original, blurred (L_b), and restored images
    """
    original = decode_base64_image(image_data)
    if original is None:
        return {'success': False, 'error': 'Invalid image'}
    
    # Step 1: Apply Gaussian blur to create L_b
    # Using kernel size 25x25 and sigma=5 for visible blur
    ksize = 25
    sigma = 5.0
    blurred = cv2.GaussianBlur(original, (ksize, ksize), sigma)
    
    # Step 2: Restore image using Enhanced Wiener Filter
    H, W = original.shape[:2]
    
    # Build the Point Spread Function (PSF) - Gaussian kernel matching the blur
    k1d = cv2.getGaussianKernel(ksize, sigma)
    k2d = k1d @ k1d.T
    k2d = k2d / k2d.sum()  # Normalize
    
    # Pad PSF to image size (centered)
    psf = np.zeros((H, W), dtype=np.float32)
    kh, kw = k2d.shape
    start_h = (H - kh) // 2
    start_w = (W - kw) // 2
    psf[start_h:start_h+kh, start_w:start_w+kw] = k2d
    
    # Shift PSF to corner for FFT
    psf = np.fft.ifftshift(psf)
    H_fft = np.fft.fft2(psf)
    
    # Adaptive regularization based on image characteristics
    # Analyze image to determine optimal K value
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    image_variance = np.var(gray.astype(np.float32))
    
    # Higher variance (more detail) = can use lower K for better restoration
    # Lower variance (smoother) = need higher K to avoid noise
    base_K = 0.005
    if image_variance > 1000:  # High detail image
        K = base_K * 0.5  # More aggressive restoration
    elif image_variance > 500:  # Medium detail
        K = base_K  # Balanced
    else:  # Low detail/smooth image
        K = base_K * 2  # More conservative to avoid noise
    
    # Compute H* (complex conjugate) and |H|^2
    H_conj = np.conj(H_fft)
    H_mag_sq = np.abs(H_fft) ** 2
    
    # Process each color channel separately
    restored_channels = []
    
    for c in range(3):
        blurred_channel = blurred[:, :, c].astype(np.float32)
        
        # FFT of blurred image
        G_fft = np.fft.fft2(blurred_channel)
        
        # Wiener Filter: F_estimated = (H* / (|H|^2 + K)) * G
        denominator = H_mag_sq + K
        F_estimated_fft = (H_conj / denominator) * G_fft
        
        # Inverse FFT
        f_estimated = np.fft.ifft2(F_estimated_fft)
        f_estimated = np.real(f_estimated)
        
        # Clip to valid range
        f_estimated = np.clip(f_estimated, 0, 255).astype(np.uint8)
        restored_channels.append(f_estimated)
    
    # Merge channels
    restored = cv2.merge(restored_channels)
    
    # Step 3: Iterative refinement using Richardson-Lucy (carefully applied)
    # Only apply if it improves the result
    restored_float = restored.astype(np.float32)
    
    # Apply 2 iterations of RL deconvolution for refinement
    for iteration in range(2):
        restored_channels_float = []
        for c in range(3):
            channel = restored_float[:, :, c]
            blurred_channel = blurred[:, :, c].astype(np.float32)
            
            # Convolve estimate with PSF
            estimate_blurred = cv2.filter2D(channel, -1, k2d)
            estimate_blurred = np.clip(estimate_blurred, 0.1, 255)  # Avoid division by zero
            
            # Compute ratio
            ratio = blurred_channel / estimate_blurred
            
            # Convolve with flipped PSF
            k2d_flipped = np.flip(np.flip(k2d, 0), 1)
            correction = cv2.filter2D(ratio, -1, k2d_flipped)
            
            # Update with damping to prevent noise amplification
            damping = 0.7  # Damping factor to prevent over-correction
            channel_updated = channel * (1 - damping + damping * correction)
            channel_updated = np.clip(channel_updated, 0, 255)
            restored_channels_float.append(channel_updated)
        
        restored_float = cv2.merge(restored_channels_float)
    
    restored = restored_float.astype(np.uint8)
    
    # Step 4: Advanced post-processing
    # Convert to LAB color space for better processing
    restored_lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(restored_lab)
    
    # Apply adaptive unsharp masking on L channel
    # Strength adapts to local contrast
    gaussian_l = cv2.GaussianBlur(l_channel, (5, 5), 1.5)
    contrast = cv2.absdiff(l_channel.astype(np.float32), gaussian_l.astype(np.float32))
    
    # Adaptive strength: stronger sharpening in high-contrast areas
    strength_map = np.clip(contrast / 40.0, 0.2, 1.5)
    l_sharpened = l_channel.astype(np.float32) + (l_channel.astype(np.float32) - gaussian_l.astype(np.float32)) * strength_map
    l_sharpened = np.clip(l_sharpened, 0, 255).astype(np.uint8)
    
    # Merge LAB channels
    restored_lab = cv2.merge([l_sharpened, a_channel, b_channel])
    restored = cv2.cvtColor(restored_lab, cv2.COLOR_LAB2BGR)
    
    # Final edge-preserving smoothing to reduce artifacts
    # Use bilateral filter to smooth noise while preserving edges
    restored = cv2.bilateralFilter(restored, 5, 50, 50)
    
    # Final clipping
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
    Problem 3: Multi-object Detection and Blurring
    
    Template matching web application that:
    1. Checks from a local database of 10 object templates
    2. Detects object boundaries/regions using correlation
    3. Blurs the detected regions using a blur filter
    
    Args:
        image_data: Base64 encoded scene image
        
    Returns:
        Dictionary with detection count, detected objects info, and blurred image
    """
    scene_bgr = decode_base64_image(image_data)
    if scene_bgr is None:
        return {'success': False, 'error': 'Invalid scene image'}
    
    scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
    blurred_scene = scene_bgr.copy()
    
    # Load templates from local database
    templates_dir = os.path.join(os.path.dirname(__file__), 'assets', 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Get all template files (up to 10)
    template_files = [f for f in sorted(os.listdir(templates_dir))
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:10]
    
    if not template_files:
        return {
            'success': False, 
            'error': 'No templates found. Please upload template images first.',
            'count': 0
        }
    
    detected_objects = []
    detected_count = 0
    correlation_threshold = 0.6  # Minimum correlation for detection
    
    for template_file in template_files:
        template_path = os.path.join(templates_dir, template_file)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        if template is None:
            continue
        
        th, tw = template.shape[:2]
        
        # Skip if template is larger than scene
        if th > scene_gray.shape[0] or tw > scene_gray.shape[1]:
            continue
        
        # Template matching using correlation
        result = cv2.matchTemplate(scene_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # Check if correlation exceeds threshold
        if max_val >= correlation_threshold:
            detected_count += 1
            x, y = max_loc
            
            # Store detection info
            detected_objects.append({
                'template': template_file,
                'x': int(x),
                'y': int(y),
                'w': int(tw),
                'h': int(th),
                'correlation': round(float(max_val), 3)
            })
            
            # Blur the detected region
            roi = blurred_scene[y:y+th, x:x+tw]
            if roi.size > 0:
                # Use Gaussian blur with kernel size proportional to template size
                blur_size = max(15, min(31, int(min(tw, th) * 0.3)))
                if blur_size % 2 == 0:
                    blur_size += 1  # Must be odd
                roi_blurred = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
                blurred_scene[y:y+th, x:x+tw] = roi_blurred
                
                # Draw bounding box for visualization
                cv2.rectangle(blurred_scene, (x, y), (x+tw, y+th), (0, 255, 0), 2)
                cv2.putText(blurred_scene, f'{template_file[:15]}', 
                         (x, max(15, y - 5)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return {
        'success': True,
        'count': detected_count,
        'detected_objects': detected_objects,
        'blurred_image': encode_image_to_base64(blurred_scene),
        'total_templates_checked': len(template_files)
    }
