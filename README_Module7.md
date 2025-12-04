# Module 7: Calibrated Stereo Vision and Pose Estimation

## Overview

This module implements calibrated stereo vision for 3D object measurement and real-time pose estimation with hand tracking. The stereo system estimates object dimensions by first computing depth (Z value) through triangulation, then calculating real-world sizes.

## Implementation Details

### Problem 1: Object Size Estimation Using Calibrated Stereo

**Stereo Calibration Pipeline:**

1. **Chessboard Detection:**
   - Pattern size configurable (default: 9×6 inner corners)
   - Sub-pixel corner refinement for accuracy
   - Multiple preprocessing methods for robust detection:
     - Standard detection
     - Adaptive thresholding
     - CLAHE contrast enhancement

2. **Stereo Calibration:**
   - Requires minimum 3 valid image pairs
   - Computes intrinsic parameters for both cameras
   - Estimates rotation matrix (R) and translation vector (T)
   - Calculates essential (E) and fundamental (F) matrices
   - Outputs baseline distance and reprojection error

3. **3D Triangulation:**
   - Undistort corresponding points
   - Construct projection matrices:
     ```
     P1 = K1 × [I | 0]
     P2 = K2 × [R | T]
     ```
   - Triangulate using `cv2.triangulatePoints`
   - Convert homogeneous to 3D coordinates

4. **Object Measurement:**
   - **Rectangular objects:** Width and length/height from corner points
   - **Circular objects:** Diameter from two edge points
   - **Polygonal objects:** All edge lengths computed

**Measurement Equations:**
```
Z = (f × B) / d
```
Where:
- Z = depth to object
- f = focal length
- B = baseline (distance between cameras)
- d = disparity (difference in x-coordinates)

### Problem 2: Uncalibrated Stereo Derivation

This problem requires hand-written mathematical derivations (not implemented in code). The derivation covers:

- Epipolar geometry fundamentals
- Fundamental matrix estimation from correspondences
- Projective reconstruction
- Self-calibration techniques
- Metric upgrade from projective reconstruction

### Problem 3: Real-Time Pose Estimation and Hand Tracking

**MediaPipe Integration:**
- Full body pose estimation (33 landmarks)
- Hand tracking (21 landmarks per hand)
- Real-time processing via webcam

**Pose Landmarks:**
- Nose, eyes, ears
- Shoulders, elbows, wrists
- Hips, knees, ankles
- Body center estimation

**Hand Landmarks:**
- Wrist
- Thumb (4 joints)
- Index finger (4 joints)
- Middle finger (4 joints)
- Ring finger (4 joints)
- Pinky (4 joints)

**Data Export:**
- CSV format with timestamp
- Landmark coordinates (x, y, z)
- Visibility scores
- Handedness (left/right)

## Testing Instructions

### Testing Stereo Calibration (Problem 1)

**Preparation:**
1. Print a chessboard pattern (9×6 inner corners recommended)
2. Measure actual square size in mm
3. Set up two cameras with fixed baseline

**Calibration:**
1. Navigate to **Module 7 → Problem 1**
2. Capture at least 3 stereo pairs of the chessboard:
   - Different angles
   - Different distances
   - Chessboard fully visible in both views
3. Enter pattern size and square dimensions
4. Click "Calibrate"
5. Note the reprojection error (should be < 1.0 pixel)

**Measurement:**
1. After calibration, capture stereo pair of target object
2. Click corresponding points on left and right images
3. Select object shape (rectangular/circular/polygon)
4. View 3D coordinates and computed dimensions

### Testing Pose Estimation (Problem 3)
1. Navigate to **Module 7 → Problem 3**
2. Allow webcam access
3. Stand in view of camera (full body visible preferred)
4. Observe real-time skeleton overlay
5. Click "Start Recording" to capture pose data
6. Export CSV file for analysis

### Testing Hand Tracking (Problem 3)
1. Navigate to **Module 7 → Problem 3**
2. Select "Hand Tracking" mode
3. Hold hands in camera view
4. Observe 21-point hand skeleton
5. Test with various gestures

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/module7/api/calibrate` | POST | Perform stereo calibration |
| `/module7/api/detect_chessboard` | POST | Detect chessboard in image pair |
| `/module7/api/triangulate` | POST | Triangulate 3D points |
| `/module7/api/measure_size` | POST | Compute object dimensions |
| `/module7/api/debug_chessboard` | POST | Debug chessboard detection |
| `/module7/api/health` | GET | Check calibration status |

## File Structure

```
modules/module7/
├── __init__.py
├── handlers.py          # Calibration, triangulation, measurement
├── routes.py            # Flask blueprint and API routes
└── templates/
    └── module7/
        ├── module7_home.html
        ├── problem1.html    # Stereo calibration interface
        ├── problem2.html    # Derivation reference
        └── problem3.html    # Pose/hand tracking interface
```

## CSV Data Format (Pose Export)

```csv
timestamp,landmark_id,landmark_name,x,y,z,visibility
1699234567.123,0,nose,0.512,0.234,0.001,0.998
1699234567.123,1,left_eye_inner,0.498,0.215,0.002,0.995
...
```

**Coordinate System:**
- x: Horizontal position (0-1, normalized)
- y: Vertical position (0-1, normalized)
- z: Depth estimate (relative)
- visibility: Confidence score (0-1)

## Calibration Tips

1. **Chessboard Quality:**
   - Print on matte paper (avoid glare)
   - Mount on rigid flat surface
   - Ensure high contrast

2. **Image Capture:**
   - Fill 50-80% of frame with pattern
   - Capture at various angles (±45°)
   - Avoid motion blur

3. **Camera Setup:**
   - Fixed baseline (measure accurately)
   - Parallel optical axes preferred
   - Consistent lighting

## Dependencies

- OpenCV (cv2) - Stereo calibration, triangulation
- NumPy - Matrix operations
- Pillow - Image handling
- MediaPipe - Pose and hand tracking

## Resources

- **Video Demo:** [Module 7 Demo Videos](https://drive.google.com/drive/folders/1lKJa6wYDAKPlGJPm28aNTb3BJQXZJwtm?usp=drive_link)
- **Derivation Document:** [Uncalibrated Stereo Reconstruction Derivation](https://docs.google.com/document/d/1xmeF3Ek7uIAratsi-S8-WAou9DPjP43YmSKWZ22JOH4/edit?usp=sharing)
- **Live Demo:** [web-production-eff5.up.railway.app](https://web-production-eff5.up.railway.app)

