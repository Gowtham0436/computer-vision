# 🚀 Simple Deployment - 3 Steps

## Step 1: Go to Render.com
1. Visit https://render.com
2. Sign up with GitHub (free)

## Step 2: Create Web Service
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Fill in these **exact** settings:

   **Name:** `computer-vision-app` (or any name you like)
   
   **Environment:** `Python 3`
   
   **Build Command:** 
   ```
   pip install -r requirements.txt && pip install gunicorn
   ```
   
   **Start Command:**
   ```
   gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120
   ```
   
   **Plan:** `Free`

## Step 3: Add One Environment Variable
1. Click **"Environment"** tab
2. Click **"Add Environment Variable"**
3. Add:
   - **Key:** `SECRET_KEY`
   - **Value:** Generate one by running this in terminal:
     ```bash
     python -c "import secrets; print(secrets.token_hex(32))"
     ```
     Copy the output and paste as the value

4. Click **"Create Web Service"**

## ✅ Done!

Your app will be live in 2-3 minutes at: `https://your-app-name.onrender.com`

**That's it!** No GitHub Actions, no API keys, no complexity. Just works! 🎉

---

## 🔄 Auto-Deploy

Every time you push to GitHub, Render automatically redeploys your app. No extra steps needed!

---

## 📝 Notes

- First load might take 30 seconds (free tier wakes up after sleep)
- Your app works exactly like local - same code, same behavior
- All your modules will work the same way

