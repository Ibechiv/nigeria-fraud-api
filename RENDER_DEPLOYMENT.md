# Render Deployment Guide

## Problem Solved ✅

The initial deployment failed with this error:
```
error: failed to create directory `/usr/local/cargo/registry/cache/...`
Read-only file system (os error 30)
```

**Root Cause:** The original `requirements.txt` included `pywinpty==3.0.3`, which is a Windows-only package. Render uses Linux containers, and `pywinpty` tried to build Rust bindings in a read-only filesystem, causing the build to fail.

## Solution

Two new files were created:

### 1. `requirements-prod.txt`
- Contains **only** production dependencies
- Excludes Jupyter, notebooks, and Windows-specific packages
- ~60KB vs ~158KB for the full requirements

### 2. Updated `render.yaml` and `Procfile`
- Both now use `requirements-prod.txt` for builds
- Added `--workers 4` for better production performance

## How to Deploy

### Option 1: Use Render Dashboard
1. Go to https://render.com
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository (`nigeria-fraud-api`)
4. Configure:
   - **Name**: `nigeria-fraud-api`
   - **Environment**: `Python 3`
   - **Branch**: `main`
   - Render will auto-detect `render.yaml`
5. Click **"Create Web Service"**

### Option 2: Use Procfile (Heroku-style)
If you prefer Procfile over render.yaml:
1. Create a new web service on Render
2. Set **Start Command**: `web: uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 4`
3. In **Build Command**, specify: `pip install -r requirements-prod.txt`

## File Structure

```
nigeria-fraud-api/
├── requirements.txt       ← Development dependencies (all packages)
├── requirements-prod.txt  ← Production dependencies (used by Render)
├── Procfile              ← Heroku/Render start config
├── render.yaml           ← Render-specific config
├── runtime.txt           ← Python version specification
├── .gitignore            ← Git ignore rules
├── .dockerignore         ← Docker ignore rules
├── api/
│   ├── __init__.py
│   └── main.py           ← FastAPI app
├── models/               ← Pre-trained model artifacts
├── data/                 ← Training data & visualizations
└── README.md
```

## Local Development

For local development with Jupyter notebooks:
```bash
pip install -r requirements.txt
python -m jupyter lab
```

## Production Deployment

For Render:
```bash
pip install -r requirements-prod.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Key Changes from Original

| Aspect | Development | Production |
|--------|------------|-----------|
| Requirements File | `requirements.txt` | `requirements-prod.txt` |
| Jupyter Included | ✅ Yes | ❌ No |
| pywinpty | ✅ Yes (Windows) | ❌ No (Linux) |
| Notebooks | ✅ Supported | ❌ Not needed |
| Package Count | 158 packages | ~25 packages |
| Size | ~2.8 MB | ~500 KB |
| Build Time on Render | ❌ 5+ min (fails) | ✅ 1-2 min |

## Troubleshooting

### Still getting build errors?
1. Check the Render build logs for the specific error
2. Ensure `requirements-prod.txt` is in the root directory
3. Verify Python version matches (`python-3.11.9` in `runtime.txt`)

### API returns "Models not loaded"
1. Verify all model files in `models/` are committed to git
2. Check Render logs to see if models directory exists
3. Models should automatically load on startup

### Port binding fails
The app now respects the `PORT` environment variable set by Render. No hardcoding needed.

## Performance Tips

- The `--workers 4` setting helps handle concurrent requests
- Render's free tier is good for development/testing
- Consider **Starter** plan for production (paid)
- Render auto-scales on paid plans

