# 🚀 Deploy to PythonAnywhere (Free Tier) - Complete Guide

This guide will help you deploy your Computer Vision application to PythonAnywhere's **free tier** with zero cost.

## 📋 Prerequisites

1. **GitHub Account** (free)
2. **PythonAnywhere Account** (free) - Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)

## 🎯 Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Prepare for PythonAnywhere deployment"
   git push origin main
   ```

2. **Note your GitHub repository URL** (e.g., `https://github.com/yourusername/computer-vision`)

### Step 2: Sign Up / Log In to PythonAnywhere

1. Go to [www.pythonanywhere.com](https://www.pythonanywhere.com)
2. Click **"Sign up for free"** or log in
3. Free tier includes:
   - 512MB disk space
   - 1 web app
   - Python 3.8, 3.9, 3.10, or 3.11
   - Your app will be at: `yourusername.pythonanywhere.com`

### Step 3: Clone Your Repository

1. **Open a Bash console** in PythonAnywhere dashboard
2. **Navigate to your home directory**:
   ```bash
   cd ~
   ```
3. **Clone your repository**:
   ```bash
   git clone https://github.com/yourusername/computer-vision.git
   cd computer-vision
   ```
   > **Note**: If your repo is private, use a Personal Access Token:
   > ```bash
   > git clone https://YOUR_TOKEN@github.com/yourusername/computer-vision.git
   > ```

### Step 4: Create Virtual Environment

1. **Create a virtual environment** (Python 3.11 recommended):
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

2. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

### Step 5: Install Dependencies (Optimized for Free Tier)

1. **Install minimal requirements** (saves disk space):
   ```bash
   pip install --user -r requirements_pythonanywhere.txt
   ```
   
   > **Note**: Using `--user` flag installs packages to your user directory to avoid permission issues.

2. **If you run out of disk space**, you can skip MediaPipe (Module 7):
   - Edit `requirements_pythonanywhere.txt` and comment out the `mediapipe` line
   - Re-run the install command

### Step 6: Configure Web App

1. **Go to Web tab** in PythonAnywhere dashboard
2. **Click "Add a new web app"**
3. **Select "Manual configuration"** (not "Flask")
4. **Choose Python 3.11**
5. **Click Next**

### Step 7: Configure WSGI File

1. **Click on the WSGI configuration file link** (usually `/var/www/yourusername_pythonanywhere_com_wsgi.py`)
2. **Delete all the default code**
3. **Replace with**:
   ```python
   import sys
   import os

   # Add your project directory to the path
   project_home = '/home/yourusername/computer-vision'
   if project_home not in sys.path:
       sys.path.insert(0, project_home)

   # Set environment variables
   os.environ['FLASK_ENV'] = 'production'
   os.environ['SECRET_KEY'] = 'your-secret-key-here-change-this'

   # Import the Flask app
   from app import create_app
   application = create_app()
   ```

4. **Replace `yourusername`** with your PythonAnywhere username
5. **Generate a secret key**:
   ```bash
   python3 -c "import secrets; print(secrets.token_hex(32))"
   ```
6. **Replace `your-secret-key-here-change-this`** with the generated key
7. **Save the file**

### Step 8: Configure Static Files and Source Code

1. **In the Web tab**, scroll down to **"Static files"** section
2. **Add static file mapping**:
   - **URL**: `/static/`
   - **Directory**: `/home/yourusername/computer-vision/static/`
3. **Add another static file mapping**:
   - **URL**: `/uploads/`
   - **Directory**: `/home/yourusername/computer-vision/uploads/`

### Step 9: Create Required Directories

1. **In Bash console**, create necessary directories:
   ```bash
   cd ~/computer-vision
   mkdir -p uploads static/outputs models
   touch uploads/.gitkeep static/outputs/.gitkeep models/.gitkeep
   ```

### Step 10: Set Up Environment Variables

1. **In Bash console**, create a `.env` file (optional, or set in WSGI file):
   ```bash
   cd ~/computer-vision
   echo "FLASK_ENV=production" > .env
   echo "SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" >> .env
   ```

### Step 11: Reload Web App

1. **Go back to Web tab**
2. **Click the green "Reload" button** (or visit `yourusername.pythonanywhere.com`)
3. **Wait 30-60 seconds** for the app to reload

### Step 12: Test Your Application

1. **Visit**: `https://yourusername.pythonanywhere.com`
2. **Test the face authentication** (models will download automatically on first use)
3. **Test Module 2 Problem 1** (template matching)

## 🔧 Troubleshooting

### Issue: "Disk quota exceeded" during pip install

**Solution**:
```bash
# Clean pip cache
rm -rf ~/.cache/pip

# Clean temporary files
rm -rf /tmp/pip-*

# Install with --no-cache-dir
pip install --user --no-cache-dir -r requirements_pythonanywhere.txt
```

### Issue: "Module not found" errors

**Solution**:
1. Check that virtual environment is activated: `source venv/bin/activate`
2. Verify packages are installed: `pip list`
3. Make sure WSGI file has correct path

### Issue: Static files not loading

**Solution**:
1. Check Static files mapping in Web tab
2. Verify directories exist: `ls -la ~/computer-vision/static/`
3. Check file permissions: `chmod -R 755 ~/computer-vision/static/`

### Issue: "Permission denied" errors

**Solution**:
```bash
# Fix permissions
chmod -R 755 ~/computer-vision
chmod -R 777 ~/computer-vision/uploads
chmod -R 777 ~/computer-vision/static/outputs
```

### Issue: App shows "Internal Server Error"

**Solution**:
1. **Check error logs** in Web tab → "Error log" link
2. **Check server log** in Web tab → "Server log" link
3. **Common fixes**:
   - Verify SECRET_KEY is set in WSGI file
   - Check that all directories exist
   - Ensure virtual environment path is correct

### Issue: Models not downloading

**Solution**:
1. Models download automatically on first use
2. Check internet connection in Bash console
3. Verify `models/` directory exists and is writable

## 📊 Disk Space Management

**Free tier limit: 512MB**

To check disk usage:
```bash
du -sh ~/*
```

**To save space**:
1. Remove `__pycache__` directories: `find ~ -type d -name __pycache__ -exec rm -r {} +`
2. Remove `.pyc` files: `find ~ -name "*.pyc" -delete`
3. Skip MediaPipe if Module 7 is not needed
4. Clean pip cache regularly: `rm -rf ~/.cache/pip`

## 🔄 Updating Your App

1. **Pull latest changes**:
   ```bash
   cd ~/computer-vision
   git pull origin main
   ```

2. **Reload web app** in Web tab

## 🎉 Success!

Your app should now be live at: `https://yourusername.pythonanywhere.com`

## 📝 Notes

- **Free tier limitations**:
  - App sleeps after 3 months of inactivity
  - 512MB disk space
  - No custom domains
  - Limited CPU time

- **Upgrading**: If you need more resources, PythonAnywhere offers paid plans starting at $5/month

## 🆘 Need Help?

- PythonAnywhere Docs: [help.pythonanywhere.com](https://help.pythonanywhere.com)
- Check error logs in Web tab
- PythonAnywhere Community: [www.pythonanywhere.com/forums](https://www.pythonanywhere.com/forums)

