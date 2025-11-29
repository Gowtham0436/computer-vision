# Fly.io Deployment Guide

## Prerequisites

- GitHub account with code pushed
- Terminal/Command line access

## Step 1: Install Fly CLI

### macOS/Linux:
```bash
curl -L https://fly.io/install.sh | sh
```

### Windows:
Download from: https://fly.io/docs/hands-on/install-flyctl/

### Verify:
```bash
fly version
```

## Step 2: Sign Up (No Credit Card!)

```bash
fly auth signup
```

This opens your browser. Sign up with email or GitHub. **No credit card required for free tier!**

## Step 3: Deploy Your App

```bash
# Navigate to your project
cd /path/to/computer-vision

# Launch (auto-detects Dockerfile)
fly launch
```

**During `fly launch`, you'll be asked:**
- App name: `computer-vision` (or your choice)
- Region: Choose closest (e.g., `iad` for US East, `sjc` for US West)
- Postgres/Redis: Say **"No"** (not needed)
- Overwrite fly.toml: Say **"Yes"** (if it exists)

## Step 4: Set Environment Variables

```bash
# Generate and set SECRET_KEY
fly secrets set SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

# Set Flask environment
fly secrets set FLASK_ENV=production
```

Or set manually:
```bash
fly secrets set SECRET_KEY=your_secret_key_here
fly secrets set FLASK_ENV=production
```

## Step 5: Deploy!

```bash
fly deploy
```

This will:
- Build your Docker image
- Push to Fly.io
- Deploy your app
- Takes 2-3 minutes

## Step 6: Open Your App!

After deployment, you'll see:
```
Visit your app at: https://your-app-name.fly.dev
```

Open that URL in your browser!

## 🔄 Updating Your App

After pushing changes to GitHub:

```bash
fly deploy
```

Or set up auto-deploy from GitHub (optional).

## 🐛 Troubleshooting

### Check logs:
```bash
fly logs
```

### Check app status:
```bash
fly status
```

### SSH into your app:
```bash
fly ssh console
```

### View app info:
```bash
fly info
```

### Restart app:
```bash
fly apps restart your-app-name
```

## 📝 Fly.io Free Tier

- **3 shared-cpu VMs**
- **3GB persistent volumes**
- **Always-on available**
- **No credit card needed**

## ✅ Post-Deployment Checklist

- [ ] App accessible at your fly.dev URL
- [ ] Face authentication works
- [ ] File uploads work
- [ ] Static files load correctly
- [ ] At least one module tested

## 🎉 Done!

Your app is live and always-on at: `https://your-app-name.fly.dev`

No disk quota issues, great performance, free tier available!

