# 🚀 Quick Deployment Steps

## Option 1: Render (Easiest - 5 minutes)

### Step 1: Create Account
1. Go to https://render.com
2. Sign up with GitHub

### Step 2: Create Web Service
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Fill in:
   - **Name**: `computer-vision-app` (or your choice)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && pip install gunicorn`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120`
   - **Plan**: Free

### Step 3: Add Environment Variables
Click **"Environment"** tab and add:
- `SECRET_KEY`: Generate with: `python -c "import secrets; print(secrets.token_hex(32))"`
- `FLASK_ENV`: `production`

### Step 4: Deploy!
Click **"Create Web Service"** - Render will deploy automatically!

Your app will be live at: `https://your-app-name.onrender.com`

---

## Option 2: Railway (Alternative)

### Step 1: Create Account
1. Go to https://railway.app
2. Sign up with GitHub

### Step 2: Create Project
1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose your repository

### Step 3: Add Environment Variables
In project settings:
- `SECRET_KEY`: Generate with: `python -c "import secrets; print(secrets.token_hex(32))"`
- `FLASK_ENV`: `production`

### Step 4: Deploy!
Railway auto-deploys on push to main branch!

---

## 🔄 Enable GitHub Actions Auto-Deploy (Optional)

### For Render:
1. Get **API Key**: Render Dashboard → Account Settings → API Keys → Create
2. Get **Service ID**: Your service → Settings → Service ID
3. Add to GitHub: Repo → Settings → Secrets → Actions
   - `RENDER_API_KEY`: Your API key
   - `RENDER_SERVICE_ID`: Your service ID

### For Railway:
1. Get **Token**: Railway Dashboard → Account Settings → Tokens → Create
2. Get **Service ID**: Your project → Settings
3. Add to GitHub Secrets:
   - `RAILWAY_TOKEN`: Your token
   - `RAILWAY_SERVICE_ID`: Your service ID

---

## ✅ Post-Deployment Checklist

- [ ] App is accessible at the URL
- [ ] Test face authentication
- [ ] Test at least one module
- [ ] Check file uploads work
- [ ] Verify static files load

---

## 🐛 Common Issues

**App won't start?**
- Check logs in platform dashboard
- Verify `SECRET_KEY` is set
- Ensure all dependencies installed

**Slow first load?**
- Free tier apps "sleep" after inactivity
- First request takes ~30 seconds to wake up
- Consider paid plan for always-on

**File uploads fail?**
- Check file size limits
- Verify `uploads/` directory exists

---

## 📝 Files Created for Deployment

- ✅ `.github/workflows/deploy.yml` - GitHub Actions for Render
- ✅ `.github/workflows/deploy-railway.yml` - GitHub Actions for Railway
- ✅ `Procfile` - Process file for Heroku/Render
- ✅ `runtime.txt` - Python version
- ✅ `render.yaml` - Render configuration
- ✅ `railway.json` - Railway configuration
- ✅ `.gitignore` - Git ignore rules
- ✅ `app.py` - Updated for production

---

## 🎉 You're Ready!

Just push these files to GitHub and follow the steps above. Your app will be live in minutes!

