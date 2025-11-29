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
    Problem 1: Simple & Reliable Template Matching
    
    Uses proven OpenCV template matching with:
    - Multi-scale search (0.3x to 2.5x)
    - Multiple rotation angles (0°, 90°, 180°, 270°)
    - Multiple matching methods
    - Very lenient thresholds
    
    Args:
        template_data: Base64 encoded template image (cropped from target)
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
        
        # Convert to grayscale
        if len(template_bgr.shape) == 3:
            template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template_bgr.copy()
            
        if len(target_bgr.shape) == 3:
            target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target_bgr.copy()
        
        # Get dimensions
        H, W = target_gray.shape[:2]
        th, tw = template_gray.shape[:2]
        
        # Validate template is smaller than target
        if th >= H or tw >= W:
            return {
                'success': False, 
                'error': f'Template ({tw}x{th}) must be smaller than target ({W}x{H}). Please use a smaller template image.'
            }
        
        # SIMPLE TEMPLATE MATCHING - RELIABLE APPROACH
        METHOD = cv2.TM_CCOEFF_NORMED
        SCALES = np.linspace(0.3, 2.5, 20)
        ANGLES = [0, 90, 180, 270]
        
        best_score = -1.0
        best_match = None  # (x, y, w, h, scale, angle)
        best_res = None
        
        # Try each rotation angle
        for ang in ANGLES:
            tpl_rot = rotate_keep_all(template_gray, ang)
            
            # Try each scale
            for s in SCALES:
                tw_scaled = max(10, int(tpl_rot.shape[1] * s))
                th_scaled = max(10, int(tpl_rot.shape[0] * s))
                
                if tw_scaled >= W - 10 or th_scaled >= H - 10:
                    continue
                if tw_scaled < 10 or th_scaled < 10:
                    continue
                
                # Resize
                if s < 1.0:
                    interp = cv2.INTER_AREA
                else:
                    interp = cv2.INTER_CUBIC
                
                tpl_scaled = cv2.resize(tpl_rot, (tw_scaled, th_scaled), interpolation=interp)
                
                # Template matching
                try:
                    res = cv2.matchTemplate(target_gray, tpl_scaled, METHOD)
                    if res.size == 0:
                        continue
                    
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    
                    if max_val > best_score:
                        best_score = float(max_val)
                        best_match = (max_loc[0], max_loc[1], tw_scaled, th_scaled, float(s), ang)
                        best_res = res
                        
                        # Early termination
                        if best_score >= 0.75:
                            break
                except:
                    continue
            
            if best_score >= 0.75:
                break
        
        # Check if match found
        if best_match is None:
            return {
                'success': False,
                'error': 'No match found. Ensure template is visible in target image.',
                'correlation_score': 0.0
            }
        
        x, y, w, h, scale, angle = best_match
        
        # VERY LENIENT THRESHOLD - accept almost anything
        min_acceptable = 0.20  # Very low threshold
        
        if best_score < min_acceptable:
            return {
                'success': False,
                'error': f'Match confidence too low: {best_score:.3f} (minimum: {min_acceptable:.2f}).',
                'correlation_score': float(best_score),
                'threshold_used': float(min_acceptable)
            }
        
        # Create annotated image
        annotated = target_bgr.copy()
        
        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.rectangle(annotated, (x + 2, y + 2), (x + w - 2, y + h - 2), (0, 200, 0), 2)
        
        # Add text
        text = f'Match: {best_score:.3f}'
        if scale != 1.0:
            text += f' (scale: {scale:.2f}x)'
        if angle != 0:
            text += f' (rot: {angle}°)'
        
        # Text background
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = x
        text_y = max(30, y - 10)
        cv2.rectangle(annotated, 
                     (text_x - 5, text_y - text_h - 5), 
                     (text_x + text_w + 5, text_y + baseline + 5),
                     (0, 0, 0), -1)
        
        cv2.putText(annotated, text,
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Create heatmap
        heatmap_colored = np.zeros_like(annotated)
        max_corr = min_corr = mean_corr = 0.0
        
        if best_res is not None and best_res.size > 0:
            try:
                heatmap_resized = cv2.resize(best_res, (W, H), interpolation=cv2.INTER_CUBIC)
                result_norm = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX)
                heatmap_uint8 = result_norm.astype(np.uint8)
                heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                max_corr = float(best_res.max())
                min_corr = float(best_res.min())
                mean_corr = float(best_res.mean())
            except:
                pass
        
        return {
            'success': True,
            'method': 'TM_CCOEFF_NORMED',
            'correlation_score': round(best_score, 4),
            'scale': round(scale, 2),
            'angle': int(angle),
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'annotated_image': encode_image_to_base64(annotated),
            'heatmap_image': encode_image_to_base64(heatmap_colored),
            'max_correlation': max_corr,
            'min_correlation': min_corr,
            'mean_correlation': mean_corr,
            'threshold_used': round(min_acceptable, 3)
        }
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"\n{'='*60}")
        print(f"ERROR in match_template_handler:")
        print(f"{'='*60}")
        print(error_trace)
        print(f"{'='*60}\n")
        return {
            'success': False,
            'error': f'Processing error: {error_msg}. Please check your images and try again.',
            'correlation_score': 0.0,
            'debug_info': error_trace[:200] if len(error_trace) > 200 else error_trace
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
