# Module 4: Image Stitching and SIFT Feature Extraction

## Overview

This module implements panoramic image stitching and SIFT (Scale-Invariant Feature Transform) feature extraction from scratch. The implementation includes RANSAC-based homography estimation and comparison with OpenCV's built-in SIFT implementation.

## Implementation Details

### Problem 1: Image Stitching Procedure

**Stitching Pipeline:**
1. Load multiple overlapping images (minimum 4 for landscape, 8 for portrait)
2. Detect features using SIFT
3. Match features between consecutive image pairs
4. Estimate homographies using RANSAC
5. Warp and blend images into panorama

**OpenCV Stitcher Integration:**
- Uses `cv2.Stitcher_create(cv2.Stitcher_PANORAMA)` mode
- Automatic exposure compensation
- Multi-band blending for seamless transitions
- Handles varying image sizes and orientations

**Reference Comparison:**
- Side-by-side display with mobile device panorama
- Visual quality assessment
- Resolution and field-of-view comparison

### Problem 2: SIFT Feature Extraction from Scratch

**Complete SIFT Implementation:**

**1. Scale-Space Extrema Detection:**
```python
# Build Gaussian pyramid with multiple octaves
num_octaves = 4
num_scales = 3
sigma = 1.6
k = 2^(1/num_scales)

# Build Difference of Gaussians (DoG) pyramid
dog_pyramid = gaussian_pyramid[i] - gaussian_pyramid[i-1]
```

**2. Keypoint Localization:**
- 3×3×3 neighborhood extrema detection
- Contrast threshold filtering (0.04)
- Edge response elimination using Hessian ratio test (threshold=10)

**3. Orientation Assignment:**
- 36-bin orientation histogram
- Gaussian-weighted gradient magnitudes
- Multiple orientations for keypoints at 80% of peak

**4. Descriptor Generation:**
- 4×4 spatial bins × 8 orientation bins = 128-dimensional vector
- Gaussian weighting from keypoint center
- Normalization and clipping (max 0.2) for illumination invariance

**RANSAC Homography Estimation:**
```python
# 4-point algorithm with 2000 iterations
# Inlier threshold: 3.0 pixels
for iteration in range(2000):
    sample = random_sample(matches, 4)
    H = compute_homography(sample)
    inliers = count_inliers(H, matches, threshold=3.0)
    if len(inliers) > best:
        best_H = H
```

**Descriptor Matching:**
- Brute-force L2 distance computation
- Lowe's ratio test (ratio=0.75)
- Cross-check validation for robustness

## Testing Instructions

### Testing Image Stitching
1. Navigate to **Module 4**
2. Capture or upload overlapping images:
   - **Landscape mode:** Minimum 4 images with 30-50% overlap
   - **Portrait mode:** Minimum 8 images with 30-50% overlap
3. Optionally upload a reference panorama (from phone's panorama feature)
4. Click "Stitch Images"
5. View:
   - Stitched panorama
   - SIFT feature matches between consecutive pairs
   - Reference comparison (if provided)

### Testing SIFT Features
1. The module automatically extracts SIFT features during stitching
2. View keypoint visualizations for each image
3. Compare custom SIFT vs OpenCV SIFT:
   - Keypoint count
   - Match count
   - Inlier count after RANSAC

### Capture Guidelines
- Maintain consistent camera height
- Overlap consecutive images by 30-50%
- Avoid moving objects in the scene
- Keep exposure consistent (manual mode recommended)
- Avoid extreme lighting variations

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/module4/api/stitch_images` | POST | Stitch images with optional reference |

**Request Body:**
```json
{
  "images": ["base64_image_1", "base64_image_2", ...],
  "reference_image": "base64_reference (optional)"
}
```

**Response:**
```json
{
  "success": true,
  "stitched_opencv": "base64_panorama",
  "sift_results": [
    {
      "pair": "1-2",
      "custom_keypoints_a": 450,
      "custom_keypoints_b": 520,
      "custom_matches": 180,
      "custom_inliers": 95,
      "opencv_inliers": 102
    }
  ],
  "sift_matches_images": ["base64_match_vis_1", ...],
  "comparison_note": "Successfully stitched 4 images..."
}
```

## File Structure

```
modules/module4/
├── __init__.py
├── handlers.py          # SIFT and stitching algorithms
├── routes.py            # Flask blueprint and API routes
└── templates/
    └── module4/
        ├── module4_home.html
        └── problem1.html    # Stitching interface
```

## Algorithm Comparison

| Feature | Custom SIFT | OpenCV SIFT |
|---------|-------------|-------------|
| Octaves | 4 | 4 |
| Scales per octave | 3 | 3 |
| Base sigma | 1.6 | 1.6 |
| Contrast threshold | 0.04 | 0.04 |
| Edge threshold | 10.0 | 10.0 |
| Descriptor size | 128 | 128 |

## Performance Considerations

- Images are resized to max 1800px width for stitching
- SIFT processing uses 960px width for speed
- RANSAC runs 2000 iterations for robust estimation
- Large panoramas may require significant memory

## Dependencies

- OpenCV (cv2) - Stitcher API, SIFT reference implementation
- NumPy - Matrix operations, SVD for homography
- Python random - RANSAC sampling

## Resources

- **Video Demo:** [Module 4 Demo Videos](https://drive.google.com/drive/folders/1m13iJuZwABcr-KuIilbYmSInasvWnB1p?usp=drive_link)
- **Live Demo:** [web-production-eff5.up.railway.app](https://web-production-eff5.up.railway.app)

