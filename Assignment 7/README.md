# Computer Vision Assignment 7

## Overview
This assignment includes implementations for stereo camera calibration & measurement (Task 1) and real-time pose estimation & hand tracking (Task 3).

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Stereo Image Pairs (Optional)
```bash
python generate_stereo_pairs_v2.py
```
This creates 3 stereo image pairs in the `stereo_pairs/` folder with:
- Pattern size: 9×6 inner corners
- Square size: 20mm
- Full-screen coverage

### 3. Start the Server
```bash
python task1_backend.py
```

The server will start on port 5001 and display:
```
============================================================
Computer Vision Assignment 7 - Server Starting
============================================================
Main page:      http://localhost:5001/
Task 1 (Stereo): http://localhost:5001/task1_stereo_measurement.html
Task 3 (Pose):   http://localhost:5001/task3_pose_hand_tracking.html
Health check:   http://localhost:5001/api/health
============================================================
```

### 4. Access the Application
Open your browser and navigate to:
**http://localhost:5001/**

## Project Structure

```
Assignment 7/
├── index.html                      # Main landing page
├── task1_stereo_measurement.html   # Task 1: Stereo calibration & measurement
├── task3_pose_hand_tracking.html   # Task 3: Pose & hand tracking
├── task1_backend.py                # Flask backend server
├── generate_stereo_pairs_v2.py     # Generate synthetic stereo images
├── requirements.txt                # Python dependencies
├── stereo_pairs/                   # Generated stereo image pairs
│   ├── left_00.jpg
│   ├── right_00.jpg
│   ├── left_01.jpg
│   ├── right_01.jpg
│   ├── left_02.jpg
│   └── right_02.jpg
└── README.md                       # This file
```

## Task 1: Stereo Calibration & Measurement

### Features
- Upload stereo image pairs (left & right cameras)
- Detect chessboard patterns automatically
- Calibrate stereo cameras using multiple image pairs
- Measure real-world object sizes through 3D triangulation
- Support for rectangular, circular, and polygon shapes

### Usage
1. Navigate to Task 1 from the main page
2. **Calibration Phase:**
   - Upload left and right calibration images
   - Click "Detect Chessboard" (need 3+ pairs for good calibration)
   - Click "Calibrate Cameras" after collecting pairs
3. **Measurement Phase:**
   - Switch to "Measure" tab
   - Upload new stereo images of objects to measure
   - Click corresponding points on left and right images
   - View 3D coordinates and measurements

### API Endpoints
- `POST /api/calibrate` - Perform stereo calibration
- `POST /api/detect_chessboard` - Detect chessboard corners
- `POST /api/triangulate` - Triangulate 3D points
- `POST /api/measure_size` - Calculate object dimensions
- `GET /api/health` - Health check

### Technologies
- **Backend:** Flask (Python)
- **Computer Vision:** OpenCV
- **Frontend:** HTML5, TailwindCSS, JavaScript

## Task 3: Pose Estimation & Hand Tracking

### Features
- Real-time pose detection (33 body landmarks)
- Real-time hand tracking (21 hand landmarks per hand, up to 2 hands)
- Frame-by-frame data recording
- CSV export of all landmark data
- Adjustable confidence thresholds
- FPS counter

### Usage
1. Navigate to Task 3 from the main page
2. Click "Start Camera" (allow camera access when prompted)
3. Enable/disable pose estimation or hand tracking
4. Adjust detection and tracking confidence sliders if needed
5. Click "Start Recording" to capture landmark data
6. Click "Stop Recording" when done
7. Click "Export to CSV" to download the data

### CSV Format
The exported CSV includes:
- Frame number and timestamp
- Pose detection status and landmark data (x, y, z, visibility)
- Hand detection status and landmark data (x, y, z)
- Handedness (Left/Right) for each detected hand

### Technologies
- **Frontend:** HTML5, TailwindCSS, JavaScript
- **Computer Vision:** MediaPipe (via CDN)
- **No backend required** - runs entirely in browser

## Requirements

### Python Packages
```
flask>=2.3.0
flask-cors>=4.0.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
```

### Browser Requirements
- Modern browser (Chrome, Edge, Firefox, Safari)
- Camera access permission (for Task 3)
- WebGL support recommended

## Notes

- **Task 1** requires the backend server to be running
- **Task 3** works standalone without backend (uses MediaPipe CDN)
- Stereo images are provided in `stereo_pairs/` folder
- Generated chessboards have 9×6 internal corners (20mm squares)
- Default calibration requires minimum 3 image pairs

## Troubleshooting

### Backend won't start
- Ensure port 5001 is not in use
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version (3.8+)

### Chessboard detection fails
- Ensure the entire 9×6 pattern is visible
- Check image quality and lighting
- Verify pattern size settings (9×6 inner corners)

### Camera not working (Task 3)
- Allow camera permission when prompted
- Try using Chrome or Edge browser
- Check camera is not in use by another application

## CSc 8830 - Advanced Computer Vision
Assignment 7

