# 🔒 Security Guide - How to Store Credentials Safely

## ✅ Three Secure Methods (None hardcoded in Python files!)

### Method 1: Streamlit Secrets File (RECOMMENDED - Local Development)

**Location:** `.streamlit/secrets.toml`

**Security Level:** 🔒🔒🔒 Very Secure
- ✅ NOT in your Python code
- ✅ NOT committed to Git (in `.gitignore`)
- ✅ Only on your local machine
- ✅ Separate from your application code

**How to use:**
1. Edit the file `.streamlit/secrets.toml`
2. Add your credentials:
   ```toml
   ERCOT_USERNAME = "your_email@example.com"
   ERCOT_PASSWORD = "your_password"
   ERCOT_CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
   ERCOT_SUBSCRIPTION_KEY = "your_subscription_key"
   ```
3. Save and restart Streamlit
4. The app will automatically detect and use these credentials

**To edit:**
```bash
nano .streamlit/secrets.toml
# or
code .streamlit/secrets.toml
```

---

### Method 2: Session-Based (In-App Entry with Memory Only)

**Security Level:** 🔒🔒 Secure
- ✅ NOT saved to disk at all
- ✅ Only stored in browser memory
- ✅ Cleared when you close the browser
- ✅ Each user enters their own credentials

**How to use:**
1. Run the app
2. Enter credentials in the sidebar form
3. Check "💾 Remember credentials for this session"
4. Credentials stay active until you close the browser

**Perfect for:** 
- Demos or presentations
- When you don't want ANY files with credentials
- When multiple people use the same computer

---

### Method 3: Streamlit Cloud Secrets (RECOMMENDED - Production/Deployment)

**Security Level:** 🔒🔒🔒🔒 Most Secure
- ✅ Encrypted in Streamlit Cloud
- ✅ Never visible in your code
- ✅ Never in your repository
- ✅ Managed through Streamlit's web interface

**How to use:**
1. Deploy your app to Streamlit Cloud
2. Go to app settings
3. Click "Secrets" in the left sidebar
4. Paste the contents of your `secrets.toml` file
5. Save

**Access:** https://share.streamlit.io/ → Your App → Settings → Secrets

---

## ❌ What NOT to Do

### DON'T do this (hardcoded in Python):
```python
# ❌ BAD - Don't do this!
username = "my_email@example.com"
password = "my_password"
```

### DON'T commit secrets to Git:
```bash
# ❌ BAD - Don't do this!
git add .streamlit/secrets.toml
```

---

## 🛡️ Current Setup

Your app now supports ALL THREE methods in this priority order:

1. **First Check:** Session state (if you entered creds in the UI and clicked "Remember")
2. **Second Check:** Environment variables (if you set `export ERCOT_USERNAME=...`)
3. **Third Check:** `.streamlit/secrets.toml` file
4. **Fallback:** Manual entry in the UI

---

## 📋 Quick Commands

### View current secrets file:
```bash
cat .streamlit/secrets.toml
```

### Edit secrets file:
```bash
nano .streamlit/secrets.toml
```

### Verify secrets file is NOT tracked by Git:
```bash
git status
# Should NOT show .streamlit/secrets.toml
```

### Check if secrets file is in .gitignore:
```bash
grep -r "secrets.toml" .gitignore
# Should show: .streamlit/secrets.toml
```

---

## ✅ Verification

After setting up, you should see in the sidebar:
- ✅ "Using credentials from Streamlit secrets file"
- 📁 "Credentials stored in `.streamlit/secrets.toml` (not in code!)"

Your credentials are now secure! 🎉
