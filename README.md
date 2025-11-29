# Computer Vision Assignment Platform

Flask web application for CSc 8830 Computer Vision assignments with multiple modules for image processing, computer vision algorithms, and real-time tracking.

## 🚀 Free Deployment Options

### Option 1: PythonAnywhere (Recommended - Easiest)

**100% Free, No Credit Card Required!**

1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Follow the complete guide: **`PYTHONANYWHERE_DEPLOY.md`**
3. Quick start: **`PYTHONANYWHERE_QUICK_START.txt`**

Your app will be live at: `https://yourusername.pythonanywhere.com`

**Perfect for free tier - optimized for 512MB disk space!**

### Option 2: Fly.io (Alternative)

### Step 1: Install Fly CLI
```bash
curl -L https://fly.io/install.sh | sh
```

### Step 2: Sign Up (No Credit Card Needed!)
```bash
fly auth signup
```

### Step 3: Deploy
```bash
# In your project directory
fly launch

# Set environment variables
fly secrets set SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
fly secrets set FLASK_ENV=production

# Deploy!
fly deploy
```

Your app will be live at: `https://your-app-name.fly.dev`

**See `DEPLOY.md` for detailed instructions.**

## 📦 Local Development

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
- `PORT`: Server port (auto-set by Fly.io)

## 🐛 Troubleshooting

- **Models not downloading?** Check internet connection, models download on first run
- **File uploads fail?** Ensure `uploads/` directory exists and is writable
- **Static files not loading?** Verify `static/` folder is in repository

## 📚 Documentation

- `DEPLOY.md` - Fly.io deployment guide
- See individual module folders for module-specific documentation

## 📄 License

Academic project for CSc 8830 Computer Vision
