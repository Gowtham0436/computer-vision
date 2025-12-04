# Module 5-6: Motion Tracking and Real-Time Object Tracking

## Overview

This module combines motion analysis and real-time object tracking implementations. It includes optical flow computation for motion estimation and three distinct tracking approaches: marker-based, markerless, and SAM2 segmentation-based tracking.

## Implementation Details

### Problem 1: Motion Tracking Derivation and Estimation

**Motion Tracking Equation:**
The fundamental brightness constancy assumption:
```
I(x + u, y + v, t + 1) = I(x, y, t)
```
Where (u, v) represents the motion vector at pixel (x, y).

**Lucas-Kanade Optical Flow Implementation:**
- Feature detection using Shi-Tomasi corners (`cv2.goodFeaturesToTrack`)
- Sparse optical flow computation (`cv2.calcOpticalFlowPyrLK`)
- Pyramidal implementation for large motions

**Dense Optical Flow (Farneback):**
- Full motion field visualization
- Parameters: pyramid scale=0.5, levels=3, window=15
- Used for motion field visualization

**Affine Motion Model (Lucas-Kanade Extension):**
```
u(x,y) = a₁x + b₁y + c₁
v(x,y) = a₂x + b₂y + c₂
```
Six parameters estimated via least squares from optical flow constraints.

### Problem 2: Real-Time Object Tracking

Three tracking implementations running in the browser using OpenCV.js:

#### (i) Marker-Based Tracking (ArUco/QR Code)

**Implementation:**
- Real-time ArUco marker detection
- Support for multiple dictionary types (4×4, 5×5, 6×6)
- Pose estimation from marker corners
- Bounding box tracking around detected markers

**Features:**
- Sub-pixel corner refinement
- ID-based marker identification
- Rotation and scale invariant detection

#### (ii) Markerless Object Tracking

**Tracking Algorithms:**
- **CAMShift (Continuously Adaptive Mean Shift):**
  - Color histogram-based tracking
  - Automatic scale and rotation adaptation
  - HSV color space for robustness

- **Template-based Tracking:**
  - Normalized cross-correlation matching
  - Multi-scale search for scale changes
  - ROI prediction for efficiency

**Initialization:**
- User selects ROI on first frame
- Histogram computed from selected region
- Back-projection for probability map

#### (iii) SAM2 Segmentation-Based Tracking

**Offline Segmentation + Real-Time Tracking:**
1. Capture reference frame
2. Generate segmentation mask using boundary detection
3. Create NPZ file containing:
   - Binary mask (1 × H × W)
   - Centroid coordinates
4. Load NPZ for real-time tracking

**NPZ File Format:**
```python
{
    'masks': np.array shape (1, H, W),  # Binary segmentation mask
    'centroids': np.array shape (1, 2)  # [cx, cy] coordinates
}
```

**Tracking with Segmentation:**
- Color histogram from masked region
- CAMShift with mask-weighted probability
- Contour tracking for boundary updates

## Testing Instructions

### Testing Motion Estimation (Problem 1)
1. Navigate to **Module 5-6 → Problem 1**
2. Upload two consecutive video frames
3. Click "Compute Motion"
4. View:
   - Sparse motion vectors (arrows on features)
   - Dense motion field
   - Mean and standard deviation of motion

### Testing Marker-Based Tracking (Problem 2.i)
1. Navigate to **Module 5-6 → Problem 2**
2. Select "Marker-Based" tracking mode
3. Print ArUco markers (6×6_250 recommended)
4. Start webcam and hold marker in view
5. Observe real-time bounding box tracking

### Testing Markerless Tracking (Problem 2.ii)
1. Navigate to **Module 5-6 → Problem 2**
2. Select "Markerless" tracking mode
3. Start webcam
4. Click and drag to select object ROI
5. Move object and observe tracking

### Testing SAM2-Based Tracking (Problem 2.iii)
1. Navigate to **Module 5-6 → Problem 2**
2. Select "SAM2" tracking mode
3. **Option A - Auto-detect:**
   - Capture frame with object
   - Click "Generate NPZ" for automatic boundary detection
4. **Option B - Manual selection:**
   - Draw ROI around object
   - Click "Create NPZ from Region"
5. Download NPZ file (optional backup)
6. Start tracking with loaded segmentation

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/module5_6/api/compute_motion` | POST | Compute optical flow between frames |
| `/module5_6/api/create_sam2_npz` | POST | Generate NPZ from auto-detected boundary |
| `/module5_6/api/create_sam2_npz_region` | POST | Generate NPZ from selected region |

## File Structure

```
modules/module5_6/
├── __init__.py
├── handlers.py          # Motion estimation, NPZ generation
├── routes.py            # Flask blueprint and API routes
└── templates/
    └── module5_6/
        ├── module5_6_home.html
        ├── problem1.html    # Motion estimation interface
        └── problem2.html    # Real-time tracking interface

static/js/module5_6/
├── app.js               # Main tracking application
├── tracker.js           # Tracking algorithm implementations
└── npz_parser.js        # NPZ file parsing utilities
```

## Client-Side Implementation

The real-time tracking runs entirely in the browser using:
- **OpenCV.js** - Computer vision operations
- **WebRTC** - Webcam access
- **Canvas API** - Visualization

This ensures responsive tracking without server round-trips.

## Performance Notes

- Tracking runs at 15-30 FPS depending on resolution
- Lower resolution (640×480) recommended for smooth tracking
- Marker detection is fastest, SAM2-based is most accurate
- Markerless tracking may drift over extended periods

## Dependencies

### Server-Side
- OpenCV (cv2) - Optical flow computation
- NumPy - Array operations, NPZ file creation
- Flask - Web framework

### Client-Side
- OpenCV.js - Browser-based computer vision
- JavaScript - Tracking logic and UI

## Resources

- **Video Demo:** [Module 5-6 Demo Videos](https://drive.google.com/drive/folders/1drMoXxM1l_9q-G7sPEzHokzbKq7n-Tz1?usp=drive_link)
- **Derivation Document:** [Lucas-Kanade & Affine Motion Model Derivation](https://docs.google.com/document/d/1LACvjROUFot_AqIzYj0mHj1TEeKIFW4FRhDNz9A23qw/edit?usp=sharing)
- **Live Demo:** [web-production-eff5.up.railway.app](https://web-production-eff5.up.railway.app)

