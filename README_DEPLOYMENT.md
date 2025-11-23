# Quick Deployment Guide

## 🚀 Fastest Way: Render (Recommended)

### 1. Sign up at [render.com](https://render.com)

### 2. Create Web Service
- Click "New +" → "Web Service"
- Connect your GitHub repository
- Use these settings:
  - **Build Command**: `pip install -r requirements.txt && pip install gunicorn`
  - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120`
  - **Plan**: Free

### 3. Add Environment Variables
In Render dashboard → Environment:
- `SECRET_KEY`: Run `python -c "import secrets; print(secrets.token_hex(32))"` and paste the result
- `FLASK_ENV`: `production`

### 4. Deploy!
Render will automatically deploy. Your app will be live at: `https://your-app-name.onrender.com`

## 🔄 Automatic Deployment with GitHub Actions

The repository includes GitHub Actions workflows that automatically deploy on push to `main` branch.

### For Render:
1. Get your Render API key: Dashboard → Account Settings → API Keys
2. Get your Service ID: Service Settings → Service ID
3. Add to GitHub Secrets (Settings → Secrets → Actions):
   - `RENDER_API_KEY`
   - `RENDER_SERVICE_ID`

### For Railway:
1. Get Railway token: Dashboard → Account Settings → Tokens
2. Get Service ID from your project
3. Add to GitHub Secrets:
   - `RAILWAY_TOKEN`
   - `RAILWAY_SERVICE_ID`

## 📝 Important Notes

- **Free tier limitations**: Render/Railway free tier may sleep after inactivity (takes ~30s to wake)
- **File uploads**: Uploaded files are stored temporarily. For production, consider cloud storage (S3, etc.)
- **Models**: Face recognition models are downloaded automatically on first run
- **HTTPS**: Both platforms provide HTTPS automatically

## 🐛 Troubleshooting

**App won't start?**
- Check logs in platform dashboard
- Verify `SECRET_KEY` is set
- Ensure all dependencies in `requirements.txt` are installed

**Static files not loading?**
- Verify `static/` folder is committed to git
- Check file paths in templates

**Sessions not persisting?**
- Ensure `SECRET_KEY` is set and consistent
- Check that cookies are enabled in browser

## 📚 Full Documentation

See `DEPLOYMENT.md` for detailed instructions for all platforms.

