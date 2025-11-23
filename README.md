# Computer Vision Assignment Platform

Flask web application for CSc 8830 Computer Vision assignments with multiple modules for image processing, computer vision algorithms, and real-time tracking.

## 🚀 Quick Deploy (No Credit Card Needed)

### Option 1: PythonAnywhere (Recommended - Always On)
1. Sign up at https://www.pythonanywhere.com (free, no card)
2. Upload code or clone from GitHub
3. Create Flask web app
4. Set WSGI to: `from app import app as application`
5. Done! See `DEPLOY.md` for details

### Option 2: Fly.io (Best Performance)
1. Install: `curl -L https://fly.io/install.sh | sh`
2. Sign up: `fly auth signup` (no card needed)
3. Deploy: `fly launch` (uses Dockerfile)
4. Done! See `DEPLOY.md` for details

### Option 3: Render (Requires Card)
1. Go to https://render.com
2. Create Web Service → Connect GitHub
3. Use Python 3 (not Docker)
4. Build: `pip install -r requirements.txt && pip install gunicorn`
5. Start: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120`
6. Add env: `SECRET_KEY` (generate with Python)
7. Done!

## 📦 Installation (Local)

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

App runs at: http://localhost:5000

## 🎯 Modules

- **Module 1**: Object Dimension Measurement (ROI & Two-Point Methods)
- **Module 2**: Template Matching, Fourier Restoration, Multi-Object Detection
- **Module 3**: Gradient Images, Edge/Corner Detection, Boundary Detection, ArUco Segmentation
- **Module 4**: Image Stitching, SIFT Feature Extraction
- **Module 5**: Motion Tracking, Object Tracking (Marker/Markerless/SAM2)
- **Module 7**: Stereo Vision, Pose & Hand Tracking

## 🔧 Requirements

- Python 3.11
- See `requirements.txt` for dependencies
- Face recognition models download automatically on first run

## 📝 Environment Variables

- `SECRET_KEY`: Flask secret key (required for production)
- `FLASK_ENV`: `production` or `development`
- `PORT`: Server port (auto-set by hosting platforms)

## 🐛 Troubleshooting

- **Models not downloading?** Check internet connection, models download on first run
- **File uploads fail?** Ensure `uploads/` directory exists and is writable
- **Static files not loading?** Verify `static/` folder is in repository

## 📚 Documentation

- `DEPLOY.md` - Detailed deployment guide for all platforms
- See individual module folders for module-specific documentation

## 📄 License

Academic project for CSc 8830 Computer Vision
