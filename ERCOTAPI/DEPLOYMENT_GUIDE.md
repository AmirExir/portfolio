# 🚀 Streamlit Cloud Deployment Guide

## Deploying Your ERCOT Dashboard to Streamlit Cloud

### Step 1: Prepare Your Repository

1. **Make sure your code is on GitHub:**
   ```bash
   git add .
   git commit -m "Add ERCOT dashboard"
   git push origin main
   ```

2. **Verify `.gitignore` includes secrets:**
   ```bash
   cat .gitignore
   # Should contain: .streamlit/secrets.toml
   ```

3. **IMPORTANT:** Never commit `secrets.toml` to GitHub!
   ```bash
   git status
   # .streamlit/secrets.toml should NOT appear
   ```

---

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Create New App:**
   - Click **"New app"** button
   - Select your repository: `AmirExir/portfolio`
   - Set branch: `main`
   - Set main file path: `ERCOTAPI/ercotapi.py`
   - Click **"Deploy"**

---

### Step 3: Add Secrets (THE IMPORTANT PART!)

This is where you enter your credentials so they're NEVER in your code!

1. **Open App Settings:**
   - After deployment starts, click on your app
   - Click the **⚙️ Menu** (three dots in top-right)
   - Select **"Settings"**

2. **Go to Secrets Section:**
   - In the left sidebar, click **"Secrets"**
   - You'll see a text editor

3. **Paste Your Credentials:**
   ```toml
   ERCOT_USERNAME = "eksir.monfared.amir@Gmail.com"
   ERCOT_PASSWORD = "your_actual_password_here"
   ERCOT_CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
   ERCOT_SUBSCRIPTION_KEY = "your_actual_subscription_key_here"
   ```

4. **Save:**
   - Click **"Save"**
   - Your app will automatically restart
   - ✅ Credentials are now encrypted and secure!

---

### Step 4: Verify Deployment

1. **Check the sidebar:**
   - Should show: ✅ "Using credentials from Streamlit secrets file"
   - If you see errors, check the logs in Streamlit Cloud

2. **Test the dashboard:**
   - Try loading data in each tab
   - Verify all features work

---

## 🔒 Security Benefits of Streamlit Cloud Secrets

✅ **Encrypted:** Credentials are encrypted at rest  
✅ **Private:** Only you (the app owner) can see/edit them  
✅ **Never in Code:** Not in your Python files or GitHub repo  
✅ **Easy Updates:** Change credentials without redeploying  
✅ **Per-App:** Different apps can have different credentials  

---

## 📝 Common Issues

### Issue: "Missing credentials" error
**Solution:** Make sure you clicked "Save" after pasting secrets

### Issue: "Authentication failed" error
**Solution:** Double-check your password and subscription key in the Secrets editor

### Issue: App won't start
**Solution:** Check the logs in Streamlit Cloud for detailed error messages

---

## 🎯 Quick Access Links

- **Streamlit Cloud Dashboard:** https://share.streamlit.io/
- **ERCOT API Portal:** https://apimarket.ercot.com/
- **Your App (after deployment):** https://share.streamlit.io/amirexir/portfolio/main/ERCOTAPI/ercotapi.py

---

## 🔄 Updating Secrets

To change your credentials after deployment:

1. Go to https://share.streamlit.io/
2. Click on your app
3. Click ⚙️ Settings → Secrets
4. Update the values
5. Click "Save"
6. App restarts automatically with new credentials

**No code changes or redeployment needed!** 🎉

---

## 📦 Local Development vs Production

| Environment | Credential Storage | How to Set |
|-------------|-------------------|------------|
| **Local Development** | `.streamlit/secrets.toml` | Edit file locally |
| **Streamlit Cloud** | Streamlit Cloud Secrets | Web interface |
| **Session Only** | Browser memory | Enter in UI |

All methods are secure and separate from your code! 🔒
