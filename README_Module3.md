# Module 3: Image Gradients, Edge/Corner Detection, and Object Segmentation

## Overview

This module implements fundamental image processing algorithms including gradient computation, edge detection, corner detection, object boundary extraction, and ArUco marker-based segmentation with SAM2 comparison. All implementations operate in real-time through a web application interface.

## Implementation Details

### Problem 1: Gradient Images and Laplacian of Gaussian

**Gradient Magnitude Computation:**
- Apply Gaussian blur (3×3, σ=0.8) for noise reduction
- Compute Sobel gradients in X and Y directions (3×3 kernel)
- Calculate magnitude: `M = √(Gx² + Gy²)`
- Normalize to 0-255 range for visualization

**Gradient Angle (Direction) Visualization:**
- HSV color-coded representation
- Hue channel encodes gradient direction (0-360°)
- Value channel encodes gradient magnitude
- Saturation set to maximum for vivid colors
- Threshold mask applied to suppress low-magnitude regions

**Laplacian of Gaussian (LoG):**
- Gaussian blur (5×5, σ=1.4) for scale-space representation
- Laplacian operator (3×3 kernel) for second-derivative computation
- Histogram equalization for enhanced visibility
- Zero-crossing detection for edge localization

### Problem 2: Edge and Corner Detection Algorithms

**Edge Detection (Canny-based):**
- Gaussian smoothing (5×5, σ=1.4)
- Automatic threshold selection using Otsu's method
- Canny edge detection with L2 gradient norm
- Morphological cleanup (dilation + closing)
- Non-maximum suppression for thin edges

**Corner Detection (Harris Algorithm):**
1. Compute image gradients Ix, Iy using Sobel operators
2. Build structure tensor components: Ixx, Iyy, Ixy
3. Apply Gaussian weighting to structure tensor (σ=1.5)
4. Compute Harris response: `R = det(M) - k × trace(M)²` where k=0.04
5. Apply relative threshold (quality × max_R)
6. Non-maximum suppression via dilation
7. Return top N corners sorted by response score

**Implementation handles:**
- Large images through automatic resizing (max 800px)
- Simple geometric shapes via contour-based fallback
- Visualization with multi-colored markers for visibility

### Problem 3: Object Boundary Detection

A generalized boundary detection system using multiple strategies:

**Detection Methods:**
1. **Edge-based (Large Morphology):** Merges nearby edges for complex objects
2. **Edge-based (Small Morphology):** Preserves fine shape details
3. **Color-based:** HSV saturation thresholding for colorful objects
4. **Adaptive Thresholding:** Works for varying illumination
5. **Otsu Thresholding:** Automatic binary segmentation

**Contour Scoring System:**
- Edge penalty (penalize contours touching image borders)
- Center score (prefer objects near image center)
- Aspect ratio validation (0.3 to 3.0 acceptable range)
- Size score (prefer medium-sized objects)
- Solidity measure (contour area / convex hull area)
- Color content analysis

### Problem 4: ArUco Marker-Based Segmentation

**Marker Detection:**
- Support for multiple ArUco dictionary types (4×4, 5×5, 6×6, 7×7)
- AprilTag format compatibility
- Robust preprocessing pipeline:
  - CLAHE contrast enhancement
  - Adaptive thresholding
  - Morphological operations
  - Multi-scale detection
  - Rotation invariant detection

**Segmentation Process:**
1. Detect all ArUco markers in image
2. Extract corner points (4 per marker)
3. Order points around centroid by angle
4. Create polygon boundary from ordered points
5. Generate binary segmentation mask

### Problem 5: SAM2 Comparison

Compares ArUco-based segmentation with SAM2 (Segment Anything Model 2):
- IoU (Intersection over Union) computation
- Side-by-side visualization (ArUco | SAM2 | Difference)
- Quantitative metrics: area comparison, difference region

## Testing Instructions

### Testing Gradient/LoG (Problem 1)
1. Navigate to **Module 3 → Problem 1**
2. Upload an image from your dataset (10 different views recommended)
3. Click "Process" to generate:
   - Gradient magnitude image
   - Gradient angle (HSV color-coded)
   - Laplacian of Gaussian filtered image
4. Compare LoG edges with gradient magnitude

### Testing Edge/Corner Detection (Problem 2)
1. Navigate to **Module 3 → Problem 2**
2. Upload test image
3. For edges: Adjust thresholds or use auto mode
4. For corners: Set max corners, quality level, min distance
5. View detected features overlaid on original

### Testing Boundary Detection (Problem 3)
1. Navigate to **Module 3 → Problem 3**
2. Upload object image
3. Click "Detect Boundary"
4. View boundary polygon and overlay visualization
5. Check area and perimeter measurements

### Testing ArUco Segmentation (Problem 4)
1. Print ArUco markers (6×6_250 dictionary recommended)
2. Attach markers around object boundary
3. Capture image from various distances and angles
4. Navigate to **Module 3 → Problem 4**
5. Upload image and view segmentation result
6. Test with at least 10 images for evaluation

### Testing SAM2 Comparison (Problem 5)
1. Generate SAM2 segmentation mask externally
2. Navigate to **Module 3 → Problem 5**
3. Upload original image and SAM2 result
4. View IoU score and visual comparison

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/module3/api/gradient_log` | POST | Compute gradients and LoG |
| `/module3/api/detect_edges` | POST | Detect edges using Canny |
| `/module3/api/detect_corners` | POST | Detect corners using Harris |
| `/module3/api/detect_boundary` | POST | Find object boundary |
| `/module3/api/segment_aruco` | POST | Segment using ArUco markers |
| `/module3/api/compare_sam2` | POST | Compare with SAM2 result |

## File Structure

```
modules/module3/
├── __init__.py
├── handlers.py          # All processing algorithms
├── routes.py            # Flask blueprint and API routes
└── templates/
    └── module3/
        ├── module3_home.html
        ├── problem1.html    # Gradient/LoG interface
        ├── problem2.html    # Edge/Corner detection
        ├── problem3.html    # Boundary detection
        ├── problem4.html    # ArUco segmentation
        └── problem5.html    # SAM2 comparison
```

## Dataset Requirements

- 10 different images of the same object
- Various angles and distances
- Consistent lighting preferred
- For ArUco: markers clearly visible with white border

## Dependencies

- OpenCV (cv2) - All image processing operations
- NumPy - Array operations, mathematical computations
- OpenCV ArUco module - Marker detection

## Resources

- **Video Demo:** [Module 3 Demo Videos](https://drive.google.com/drive/folders/1UHn1uRgJSzDtCxkuvjkzxLLVsO_vYNva?usp=drive_link)
- **Live Demo:** [web-production-eff5.up.railway.app](https://web-production-eff5.up.railway.app)

