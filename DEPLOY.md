# Deployment Guide

## 🆓 Free Hosting (No Credit Card)

### PythonAnywhere (Recommended)
1. Sign up: https://www.pythonanywhere.com
2. Go to "Web" → "Add a new web app"
3. Choose Flask → Python 3.11
4. Upload code or clone: `git clone https://github.com/YOUR_USERNAME/computer-vision.git`
5. Install: `pip3.11 install --user -r requirements.txt`
6. Edit WSGI file:
   ```python
   import sys
   project_home = '/home/YOUR_USERNAME/computer-vision'
   sys.path.insert(0, project_home)
   from app import app as application
   ```
7. Set working directory: `/home/YOUR_USERNAME/computer-vision`
8. Add env vars: `SECRET_KEY` (generate with Python)
9. Reload web app
10. Done! Always-on, no sleep!

### Fly.io
1. Install: `curl -L https://fly.io/install.sh | sh`
2. Sign up: `fly auth signup` (no card)
3. Deploy: `fly launch` (auto-detects Dockerfile)
4. Set secrets: `fly secrets set SECRET_KEY=your_key`
5. Deploy: `fly deploy`
6. Done! Always-on!

## 💳 Render (Requires Card)

1. Go to https://render.com
2. New → Web Service → Connect GitHub
3. Settings:
   - **Language**: Python 3 (NOT Docker)
   - **Build**: `pip install -r requirements.txt && pip install gunicorn`
   - **Start**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120`
4. Add env: `SECRET_KEY`
5. Create service
6. Done! (Note: Free tier sleeps after 15min inactivity)

## 🔑 Generate SECRET_KEY

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## 📝 Notes

- **PythonAnywhere**: Best for always-on, no card needed
- **Fly.io**: Best performance, Docker-ready
- **Render**: Easiest setup, but requires card and may sleep

