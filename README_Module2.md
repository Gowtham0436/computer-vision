# Module 2: Template Matching and Fourier Transform Image Restoration

## Overview

This module implements object detection using **template matching through correlation** and image restoration using **Fourier Transform** techniques. The implementation includes a multi-object detection system with automatic blurring of detected regions.

## Implementation Details

### Problem 1: Object Detection Using Template Matching (Correlation Method)

The template matching implementation uses **Normalized Cross-Correlation (TM_CCOEFF_NORMED)** with the following enhancements:

**Multi-Scale Template Matching:**
- Searches across 25 scale levels (0.3× to 2.0×)
- Handles objects appearing at different sizes than the template
- Automatic scale selection based on correlation score

**Feature-Based Correlation (Hybrid Approach):**
- SIFT feature extraction for robust keypoint detection
- FLANN-based matcher for efficient descriptor matching
- Lowe's ratio test (0.75 threshold) for match filtering
- Homography estimation using RANSAC for geometric verification
- Falls back to pure template matching when feature matching fails

**Key Implementation:**
```python
# Correlation method using cv2.matchTemplate
result = cv2.matchTemplate(target_gray, resized_template, cv2.TM_CCOEFF_NORMED)
```

### Problem 2: Image Restoration Using Fourier Transform

The restoration pipeline implements **Wiener Deconvolution** in the frequency domain:

1. **Gaussian Blur Application:** Apply known blur kernel (51×51, σ=12) to create degraded image L_b
2. **PSF Construction:** Build Point Spread Function matching the blur kernel
3. **Wiener Filter:** Apply inverse filtering with regularization parameter K=0.005
4. **Frequency-Domain Sharpening:** High-pass emphasis filter for edge enhancement
5. **Post-Processing:** Bilateral filtering for noise reduction

**Wiener Filter Equation:**
```
F_restored = (PSF_conj / (|PSF|² + K)) × G
```
Where G is the FFT of the blurred image.

### Problem 3: Multi-Object Detection and Blurring

A template matching web application that:
- Maintains a local database of up to 10 object templates
- Detects multiple objects in a scene using correlation matching
- Applies Gaussian blur to detected regions
- Handles overlapping detections through confidence-based filtering

## Testing Instructions

### Testing Template Matching (Problem 1)
1. Navigate to **Module 2 → Problem 1**
2. Upload a template image (cropped object from a different scene)
3. Upload the target/scene image containing the object
4. Adjust correlation threshold (default: 0.35)
5. Click "Match" to detect the object
6. View correlation heatmap and detection results

### Testing Fourier Restoration (Problem 2)
1. Navigate to **Module 2 → Problem 2**
2. Upload an original image
3. Click "Process" to see:
   - Original image (L)
   - Blurred image (L_b)
   - Restored image using Fourier Transform

### Testing Multi-Object Detection (Problem 3)
1. Navigate to **Module 2 → Problem 3**
2. Upload template images to the database (up to 10)
3. Upload a scene image
4. Click "Detect & Blur" to find and blur all matching objects
5. View detection count and confidence scores

### Evaluation Requirements
- Test with 10 different objects across same or different images
- Template must be from a completely different scene than the test image
- Document correlation scores for each detection

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/module2/api/match_template` | POST | Perform template matching |
| `/module2/api/restore_image` | POST | Apply Fourier-based restoration |
| `/module2/api/detect_blur` | POST | Detect and blur multiple objects |
| `/module2/api/upload_template` | POST | Add template to database |
| `/module2/api/list_templates` | GET | List all stored templates |
| `/module2/api/delete_template` | POST | Remove template from database |

## File Structure

```
modules/module2/
├── __init__.py
├── handlers.py          # Core algorithms
├── routes.py            # Flask blueprint and API routes
├── assets/
│   └── templates/       # Stored template images
└── templates/
    └── module2/
        ├── module2_home.html
        ├── problem1.html    # Template matching interface
        ├── problem2.html    # Fourier restoration interface
        └── problem3.html    # Multi-object detection interface
```

## Algorithm Details

### Correlation Matching Pipeline
1. Convert images to grayscale
2. Apply multi-scale search (25 scales)
3. Compute normalized cross-correlation at each scale
4. Select best match above threshold
5. Return bounding box and confidence score

### SIFT-Enhanced Detection
1. Extract SIFT keypoints and descriptors
2. Match descriptors using FLANN with KNN (k=2)
3. Apply Lowe's ratio test
4. Compute homography with RANSAC (2000 iterations, 3.0px threshold)
5. Transform template corners to find object location

## Dependencies

- OpenCV (cv2) - Template matching, SIFT, image processing
- NumPy - FFT operations, array manipulation
- SciPy - Signal processing utilities

## Resources

- **Video Demo:** [Module 2 Demo Videos](https://drive.google.com/drive/folders/1iy3I0BbV6KJRbN1B4AM7Tou4ehtyGQgp?usp=drive_link)
- **Live Demo:** [web-production-eff5.up.railway.app](https://web-production-eff5.up.railway.app)

