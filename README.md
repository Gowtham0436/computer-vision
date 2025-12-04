# CSc 8830 Computer Vision Project

A comprehensive web-based computer vision application implementing fundamental algorithms for object measurement, template matching, image processing, feature extraction, motion tracking, stereo vision, and pose estimation.

## Project Structure

```
computer-vision/
├── app.py                    # Main Flask application
├── wsgi.py                   # WSGI entry point for production
├── requirements.txt          # Python dependencies
├── core/                     # Shared utilities
│   ├── auth.py              # Face authentication
│   ├── decorators.py        # Route decorators
│   └── utils.py             # Image processing utilities
├── modules/                  # Feature modules
│   ├── module1/             # Object dimension measurement
│   ├── module2/             # Template matching & Fourier transform
│   ├── module3/             # Edge/corner detection & segmentation
│   ├── module4/             # Image stitching & SIFT
│   ├── module5_6/           # Motion tracking
│   └── module7/             # Stereo vision & pose estimation
├── static/                   # Static assets
│   ├── js/                  # Client-side JavaScript
│   └── outputs/             # Generated output images
├── templates/               # HTML templates
├── models/                  # Pre-trained models (face detection)
└── uploads/                 # Temporary upload storage
```

## Modules Overview

| Module | Description | Documentation |
|--------|-------------|---------------|
| **Module 1** | Real-world object dimension measurement using perspective projection | [README_Module1.md](README_Module1.md) |
| **Module 2** | Template matching (correlation), Fourier transform image restoration | [README_Module2.md](README_Module2.md) |
| **Module 3** | Gradients, edge/corner detection, boundary detection, ArUco segmentation | [README_Module3.md](README_Module3.md) |
| **Module 4** | Panoramic image stitching, SIFT feature extraction from scratch | [README_Module4.md](README_Module4.md) |
| **Module 5-6** | Optical flow motion estimation, real-time object tracking | [README_Module5_6.md](README_Module5_6.md) |
| **Module 7** | Calibrated stereo measurement, pose estimation, hand tracking | [README_Module7.md](README_Module7.md) |

---

## Local Development Setup

### Prerequisites

- Python 3.11 (recommended for best compatibility)
- pip (Python package manager)
- Webcam (for real-time features)
- Modern web browser (Chrome, Firefox, Safari)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/computer-vision.git
   cd computer-vision
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Create virtual environment
   python3.11 -m venv cvenv

   # Activate (macOS/Linux)
   source cvenv/bin/activate

   # Activate (Windows)
   cvenv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set environment variables (optional for development):**
   ```bash
   export FLASK_ENV=development
   export SECRET_KEY=your-secret-key-here
   ```

5. **Run the development server:**
   ```bash
   python app.py
   ```

6. **Access the application:**
   Open `http://localhost:5000` in your browser.

### Running Tests

```bash
# Test individual modules
python -c "from modules.module1.handlers import calculate_roi_dimensions; print('Module 1 OK')"
python -c "from modules.module2.handlers import match_template_handler; print('Module 2 OK')"
python -c "from modules.module3.handlers import detect_edges_handler; print('Module 3 OK')"
```

---

## Deployment Guide

### Option 1: Render (Recommended)

1. Create account at [render.com](https://render.com)
2. Connect your GitHub repository
3. Create new Web Service with these settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn wsgi:app`
   - **Environment Variables:**
     - `SECRET_KEY`: Generate a secure random string
     - `FLASK_ENV`: `production`
     - `PYTHON_VERSION`: `3.11.0`

### Option 2: Railway

1. Create account at [railway.app](https://railway.app)
2. Connect GitHub repository
3. Railway auto-detects Python and uses `railway.json` configuration
4. Set environment variables in dashboard

### Option 3: PythonAnywhere

1. Create account at [pythonanywhere.com](https://pythonanywhere.com)
2. Upload code or clone from GitHub
3. Create virtual environment:
   ```bash
   mkvirtualenv --python=/usr/bin/python3.11 cvenv
   pip install -r requirements_pythonanywhere.txt
   ```
4. Configure WSGI file to point to `wsgi.py`

### Option 4: Docker

```bash
# Build image
docker build -t computer-vision .

# Run container
docker run -p 5000:5000 -e SECRET_KEY=your-secret computer-vision
```

### Environment Variables for Production

| Variable | Description | Required |
|----------|-------------|----------|
| `SECRET_KEY` | Flask session encryption key | Yes |
| `FLASK_ENV` | Set to `production` | Yes |
| `PORT` | Server port (default: 5000) | No |

---

## ⚠️ Important Notice: Hosted Deployment Limitations

> **The code and implementations work perfectly on local machines but may experience issues on freely hosted websites due to limitations in processing power, storage, and RAM.**

### Known Limitations on Free Hosting Tiers:

1. **Memory Constraints:**
   - Image stitching with multiple large images may fail
   - SIFT feature extraction on high-resolution images may timeout
   - Stereo calibration with many image pairs may exceed memory

2. **Processing Time:**
   - Complex operations may timeout (30-60 second limits)
   - Real-time features require client-side processing

3. **Storage:**
   - Temporary files may be cleared between requests
   - Large file uploads may be rejected

4. **Recommendations:**
   - Use paid tiers for production deployment
   - Reduce image resolution before upload
   - Process fewer images at once for stitching
   - Run computationally intensive tasks locally

---

## Quick Links

### Live Demo
**Publicly Accessible Website:** [https://web-production-eff5.up.railway.app](https://web-production-eff5.up.railway.app)

### All Resources at a Glance

| Resource | Link |
|----------|------|
| **Live Website** | https://web-production-eff5.up.railway.app |
| **All Video Demos (Root Folder)** | https://drive.google.com/drive/folders/1sDuxKY6cSGA5d_JrrPNI5vKIN9wEUTt5?usp=drive_link |

---

## Video Demonstrations

All video demonstrations are organized by module in Google Drive:

| Module | Description | Video Demo Link |
|--------|-------------|-----------------|
| **Module 1** | Object Dimension Measurement | https://drive.google.com/drive/folders/12DCsXSPZt4a09L4Dp8GqmWm7n2oCG9VD?usp=sharing |
| **Module 2** | Template Matching & Fourier Transform | https://drive.google.com/drive/folders/1iy3I0BbV6KJRbN1B4AM7Tou4ehtyGQgp?usp=drive_link |
| **Module 3** | Edge/Corner Detection & Segmentation | https://drive.google.com/drive/folders/1UHn1uRgJSzDtCxkuvjkzxLLVsO_vYNva?usp=drive_link |
| **Module 4** | Image Stitching & SIFT | https://drive.google.com/drive/folders/1m13iJuZwABcr-KuIilbYmSInasvWnB1p?usp=drive_link |
| **Module 5-6** | Motion & Object Tracking | https://drive.google.com/drive/folders/1drMoXxM1l_9q-G7sPEzHokzbKq7n-Tz1?usp=drive_link |
| **Module 7** | Stereo Vision & Pose Estimation | https://drive.google.com/drive/folders/1lKJa6wYDAKPlGJPm28aNTb3BJQXZJwtm?usp=drive_link |

---

## Derivation Documents

Theoretical derivations and mathematical proofs (handwritten/typed documents):

| Module | Problem | Derivation Document Link |
|--------|---------|--------------------------|
| **Module 1** | Perspective Projection Equations | https://docs.google.com/document/d/1LCS-pMZc43crUcNrvYt7jO-VWAmUCUcfcD0CZ9NHdmw/edit?usp=sharing |
| **Module 5-6** | Lucas-Kanade Optical Flow & Affine Motion Model | https://docs.google.com/document/d/1LACvjROUFot_AqIzYj0mHj1TEeKIFW4FRhDNz9A23qw/edit?usp=sharing |
| **Module 7** | Uncalibrated Stereo Reconstruction | https://docs.google.com/document/d/1xmeF3Ek7uIAratsi-S8-WAou9DPjP43YmSKWZ22JOH4/edit?usp=sharing |

---

## Technology Stack

### Backend
- **Flask 3.0** - Web framework
- **OpenCV 4.8** - Computer vision operations
- **NumPy** - Numerical computations
- **MediaPipe** - Pose and hand tracking
- **Gunicorn** - Production WSGI server

### Frontend
- **HTML5/CSS3** - User interface
- **JavaScript** - Client-side logic
- **OpenCV.js** - Browser-based CV operations
- **Canvas API** - Image rendering

### Authentication
- **Face Recognition** - OpenCV DNN-based face detection and recognition
- **Session Management** - Flask sessions with 24-hour persistence

---

## API Reference

All modules expose RESTful APIs. See individual module READMEs for detailed endpoint documentation.

**Common Response Format:**
```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Error description"
}
```

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

---

## License

This project is developed for educational purposes as part of CSc 8830 coursework.

---

## Acknowledgments

- OpenCV community for comprehensive computer vision library
- MediaPipe team for pose estimation models
- Flask community for the web framework

