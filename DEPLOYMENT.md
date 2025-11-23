# Deployment Guide

This guide explains how to deploy the Computer Vision Flask application to various platforms using GitHub Actions.

## Prerequisites

1. GitHub repository with your code pushed
2. Account on one of the deployment platforms (Render, Railway, or Fly.io)

## Option 1: Deploy to Render (Recommended - Free Tier Available)

### Step 1: Create Render Account
1. Go to [render.com](https://render.com) and sign up
2. Connect your GitHub account

### Step 2: Create Web Service
1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Configure:
   - **Name**: computer-vision-app (or your choice)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120`
   - **Plan**: Free (or paid if needed)

### Step 3: Set Environment Variables
In Render dashboard, add these environment variables:
- `SECRET_KEY`: Generate a secure random key (use: `python -c "import secrets; print(secrets.token_hex(32))"`)
- `FLASK_ENV`: `production`
- `PORT`: (Render sets this automatically)

### Step 4: Get Render API Credentials
1. Go to Render Dashboard → Account Settings → API Keys
2. Create a new API key
3. Note your Service ID (found in service settings)

### Step 5: Add GitHub Secrets
1. Go to your GitHub repo → Settings → Secrets and variables → Actions
2. Add these secrets:
   - `RENDER_API_KEY`: Your Render API key
   - `RENDER_SERVICE_ID`: Your Render service ID

### Step 6: Enable GitHub Actions
The workflow file `.github/workflows/deploy.yml` is already configured. It will automatically deploy on push to `main` branch.

## Option 2: Deploy to Railway

### Step 1: Create Railway Account
1. Go to [railway.app](https://railway.app) and sign up
2. Connect your GitHub account

### Step 2: Create New Project
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your repository

### Step 3: Configure Environment Variables
Add in Railway dashboard:
- `SECRET_KEY`: Generate a secure random key
- `FLASK_ENV`: `production`
- `PORT`: (Railway sets this automatically)

### Step 4: Get Railway Token
1. Go to Railway Dashboard → Account Settings → Tokens
2. Create a new token
3. Note your Service ID

### Step 5: Add GitHub Secrets
Add to GitHub Secrets:
- `RAILWAY_TOKEN`: Your Railway token
- `RAILWAY_SERVICE_ID`: Your Railway service ID

### Step 6: Use Railway Workflow
The workflow file `.github/workflows/deploy-railway.yml` is configured. Enable it if using Railway.

## Option 3: Deploy to Fly.io

### Step 1: Install Fly CLI
```bash
curl -L https://fly.io/install.sh | sh
```

### Step 2: Create Fly.io App
```bash
fly launch
```

### Step 3: Deploy
```bash
fly deploy
```

## Manual Deployment (Without GitHub Actions)

### Using Render
1. Connect your GitHub repo in Render dashboard
2. Render will auto-deploy on every push to main branch

### Using Railway
1. Connect your GitHub repo in Railway dashboard
2. Railway will auto-deploy on every push

## Post-Deployment Checklist

- [ ] Verify the app is accessible at the provided URL
- [ ] Test face authentication
- [ ] Test at least one module functionality
- [ ] Check that file uploads work
- [ ] Verify static files are served correctly
- [ ] Test session persistence

## Troubleshooting

### Application won't start
- Check logs in your platform's dashboard
- Verify all environment variables are set
- Ensure `requirements.txt` includes all dependencies
- Check that `PORT` environment variable is set (platforms usually set this)

### Static files not loading
- Verify `static/` folder is in repository
- Check that paths in templates use `/static/...`

### File uploads not working
- Ensure `uploads/` directory exists and is writable
- Check `MAX_CONTENT_LENGTH` setting
- Verify file size limits

### Session issues
- Ensure `SECRET_KEY` is set and consistent
- Check `SESSION_COOKIE_SECURE` setting (should be True with HTTPS)

## Environment Variables Reference

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `SECRET_KEY` | Flask secret key for sessions | Yes | (none) |
| `FLASK_ENV` | Flask environment (development/production) | No | development |
| `PORT` | Port to run application | No | 5000 |

## Notes

- The free tier on Render/Railway may have limitations (sleep after inactivity, resource limits)
- For production use, consider paid plans for better performance
- Always use HTTPS in production (platforms provide this automatically)
- Keep your `SECRET_KEY` secret and never commit it to the repository

