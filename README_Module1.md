# Module 1: Real-World Object Dimension Measurement

## Overview

This module implements a web-based application for computing real-world dimensions of objects using **perspective projection equations**. The system allows users to capture or upload images and measure object dimensions (width, height, diameter) by selecting regions of interest or clicking on specific points.

## Implementation Details

### Problem 1: Perspective Projection-Based Measurement Script

The core measurement algorithm uses the **pinhole camera model** and perspective projection equations:

```
X_real = (x_pixel - cx) × Z / fx
Y_real = (y_pixel - cy) × Z / fy
```

Where:
- `(x_pixel, y_pixel)` - Image coordinates of the point
- `(cx, cy)` - Principal point (optical center)
- `(fx, fy)` - Focal lengths in pixels
- `Z` - Distance from camera to object plane (in mm)
- `(X_real, Y_real)` - Real-world coordinates

**Key Implementation Features:**
- Automatic scaling of camera intrinsic parameters based on image resolution
- Support for both portrait and landscape orientations
- Calibrated for iPhone camera (4284×5712 reference resolution) with automatic adaptation to any device
- Unit conversion between millimeters and inches

### Problem 2: Web Application Implementation

Two measurement methods are provided:

#### ROI Selection Method
- Users draw a bounding box around the object
- System calculates width and height from the selected region
- Visual annotations overlay the measured dimensions on the image

#### Two-Point Click Method
- Users click two points on the object
- System calculates the Euclidean distance between points in real-world coordinates
- Supports measuring diagonal dimensions and arbitrary distances

### Camera Parameters

Default calibration values (adjustable via UI):
- **Focal Length (fx, fy):** 5200 pixels (at reference resolution)
- **Principal Point (cx, cy):** Image center
- **Object Distance (Z):** User-specified (typically 300-1000mm)

## Testing Instructions

1. Navigate to **Module 1** from the main menu
2. Select either **Problem 1 (ROI Method)** or **Problem 2 (Two-Point Method)**
3. Upload an image or capture using webcam
4. Set the camera parameters:
   - Enter the measured distance from camera to object (Z value in mm)
   - Adjust focal length if using a different camera
5. For ROI Method: Draw a rectangle around the object
6. For Two-Point Method: Click on two points of interest
7. View the calculated dimensions in both mm and inches

### Validation Experiment

To validate accuracy:
1. Place an object with known dimensions at a measured distance
2. Capture the image ensuring the object is parallel to the image plane
3. Compare measured vs actual dimensions
4. Expected accuracy: ±5% for objects at 300-500mm distance

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/module1/api/calculate_roi` | POST | Calculate dimensions from ROI selection |
| `/module1/api/calculate_points` | POST | Calculate dimensions from two points |
| `/module1/api/evaluation` | GET | Retrieve evaluation metrics for all test objects |

## File Structure

```
modules/module1/
├── __init__.py
├── handlers.py          # Core measurement algorithms
├── routes.py            # Flask blueprint and API routes
├── evaluation_data.py   # Pre-measured evaluation dataset
└── templates/
    └── module1/
        ├── module1_home.html
        ├── problem1.html    # ROI selection interface
        └── problem2.html    # Two-point click interface
```

## Evaluation Dataset

The module includes an evaluation dataset with 12 objects measured at various distances. Physical dimensions are compared against computed measurements to assess accuracy.

## Dependencies

- OpenCV (cv2) - Image processing and annotation
- NumPy - Numerical computations
- Flask - Web framework

## Resources

- **Video Demo:** [Module 1 Demo Videos](https://drive.google.com/drive/folders/12DCsXSPZt4a09L4Dp8GqmWm7n2oCG9VD?usp=sharing)
- **Derivation Document:** [Perspective Projection Derivation](https://docs.google.com/document/d/1LCS-pMZc43crUcNrvYt7jO-VWAmUCUcfcD0CZ9NHdmw/edit?usp=sharing)
- **Live Demo:** [web-production-eff5.up.railway.app](https://web-production-eff5.up.railway.app)

