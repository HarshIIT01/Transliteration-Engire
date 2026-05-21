# Chrome Extension — Transliteration Engine

## Overview

The Chrome extension provides a browser-native interface for the Transliteration Engine. It runs as a **popup extension** — click the icon in the toolbar, type Roman text, and get instant Devanagari output.

### Architecture

```
┌─────────────────────────────┐         ┌───────────────────────┐
│  Chrome Extension (popup)   │  HTTP   │  FastAPI Server       │
│  ─────────────────────────  │ ◄─────► │  ───────────────────  │
│  popup.html / popup.js      │  POST   │  localhost:8000       │
│  Dark glassmorphism UI      │         │  /transliterate       │
│  chrome.storage.local       │         │  Seq2Seq + Attention  │
└─────────────────────────────┘         └───────────────────────┘
              ▲                                    ▲
              │                                    │
        Manifest V3                     TransliterationSystem
        No background scripts           (model + tokenizer +
        Minimal permissions              dictionary backoff)
```

**Data flow:**
1. User types Roman text in the popup textarea
2. `popup.js` sends a POST request to `localhost:8000/transliterate`
3. FastAPI server runs inference through the neural model
4. Response (Devanagari string + latency) is displayed in the popup
5. Translation is saved to history via `chrome.storage.local`

---

## File Structure

```
extension/
├── manifest.json     # Chrome Manifest V3 configuration
├── popup.html        # Popup UI (input, output, history)
├── popup.css         # Dark glassmorphism styling
├── popup.js          # API calls, clipboard, history logic
└── icon.png          # Extension icon (128×128)
```

---

## Setup & Installation

### Prerequisites
- FastAPI server running: `python -m uvicorn app.main:app`
- Google Chrome or Chromium-based browser

### Steps
1. Open Chrome → navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in top-right)
3. Click **"Load unpacked"**
4. Select the `extension/` folder
5. The extension icon appears in the toolbar
6. Click it → type text → transliterate!

> **Note:** The server must be running at `localhost:8000` for the extension to work. The green/red dot in the header shows server status.

---

## Features

| Feature | Implementation |
|---------|---------------|
| **Real-time status** | Health check every 10s via `GET /health` — green/red dot |
| **Transliteration** | `POST /transliterate` with `{"text": "..."}` |
| **Latency display** | Shows both server-side and round-trip latency |
| **Copy to clipboard** | `navigator.clipboard.writeText()` with fallback |
| **History** | Last 10 translations saved in `chrome.storage.local` |
| **Keyboard shortcut** | `Ctrl+Enter` to transliterate |
| **Click-to-recall** | Click any history item to re-populate input/output |

---

## Technical Details

### Manifest V3

Chrome extensions must use Manifest V3 (V2 is deprecated). Key decisions:

```json
{
  "manifest_version": 3,
  "permissions": [],
  "host_permissions": ["http://localhost:8000/*"]
}
```

- **No background script** — the extension is purely a popup (service workers are unnecessary for this use case)
- **Minimal permissions** — only `host_permissions` for localhost; no `tabs`, `activeTab`, or `storage` permissions needed since `chrome.storage.local` is available by default
- **No content scripts** — we don't inject into web pages

### CORS

The Chrome extension runs from `chrome-extension://<ID>` origin, which is different from `localhost:8000`. The FastAPI server includes CORS middleware:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

Without this, the browser blocks cross-origin requests from the extension popup.

### Storage

History is persisted using `chrome.storage.local`:
```javascript
chrome.storage.local.set({ history: [...entries] });
chrome.storage.local.get("history", (result) => { ... });
```

When running outside Chrome (e.g., opening `popup.html` directly in a browser for testing), the code falls back to `localStorage`.

---

## Limitations

| Limitation | Explanation |
|------------|-------------|
| **Requires server** | The extension calls `localhost:8000` — if the FastAPI server isn't running, it shows a red status dot and disables the button |
| **Not publishable to Chrome Web Store** | Localhost-dependent extensions are rejected. Publishing would require hosting the server (e.g., on a cloud VM) |
| **Not truly on-device** | The neural model runs on the server, not in-browser. True on-device would require ONNX Runtime Web (~5MB overhead) |
| **Single language** | Currently supports only Hindi (Devanagari). The architecture supports other Indic scripts with retraining |

---

## Future Improvements

1. **ONNX Runtime Web** — Run inference directly in the browser using the exported `encoder.onnx` + `decoder.onnx` files. This would make the extension fully offline.
2. **Content script injection** — Add a floating transliteration button next to text inputs on any website (Gmail, WhatsApp Web, etc.)
3. **Auto-detect intent** — Automatically detect when the user is typing in Roman Hindi and offer transliteration
4. **Multi-language support** — Extend to Tamil, Bengali, Telugu, etc.

---

## API Reference

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /transliterate`
**Request:**
```json
{ "text": "namaste doston" }
```

**Response:**
```json
{
  "input": "namaste doston",
  "result": "नमस्ते दोस्तों",
  "latency_ms": 3.42
}
```

---

## Cloud Deployment Guide

Deploying the FastAPI server to a cloud platform makes the extension work without running a local server. Once deployed, you update one line in `popup.js` and the extension points to your cloud URL.

### Recommended Platform: Render (Free Tier)

[Render](https://render.com) offers a free web service tier with:
- 512 MB RAM (enough for our 4MB model)
- Auto-deploy from GitHub
- Free HTTPS URL (`https://your-app.onrender.com`)
- Native Python support (no Docker needed)

### Step 1: Create `requirements.txt`

Create at the project root:

```
torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pandas>=2.0.0
tokenizers>=0.13.0
jinja2>=3.1.0
```

> **Note:** The `--index-url .../cpu` flag installs CPU-only PyTorch (~200MB instead of ~2GB with CUDA).

### Step 2: Create `render.yaml` (optional, for one-click deploy)

```yaml
services:
  - type: web
    name: transliteration-engine
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: "3.11"
```

### Step 3: Deploy on Render

1. Push your project to GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect your GitHub repo
4. Configure:
   - **Name:** `transliteration-engine`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
5. Click **Create Web Service**
6. Wait for build + deploy (~5 minutes)
7. Your URL: `https://transliteration-engine.onrender.com`

### Step 4: Update Extension to Use Cloud URL

In `extension/popup.js`, change **one line**:

```javascript
// Before (localhost)
const API_BASE = "http://localhost:8000";

// After (deployed)
const API_BASE = "https://transliteration-engine.onrender.com";
```

Also update `extension/manifest.json`:

```json
"host_permissions": [
    "https://transliteration-engine.onrender.com/*"
]
```

Reload the extension in `chrome://extensions` → done!

---

## Keeping the Server Warm (Cron Jobs)

Free-tier cloud services **spin down after 15 minutes of inactivity**. The first request after spin-down takes ~30 seconds (cold start: loading model + tokenizer). Solutions:

### Option A: Cron-job.org (Free, Recommended)

1. Go to [cron-job.org](https://cron-job.org)
2. Create account → **New Cron Job**
3. Configure:
   - **URL:** `https://transliteration-engine.onrender.com/health`
   - **Schedule:** Every 14 minutes (`*/14 * * * *`)
   - **Method:** GET
4. Save

This pings `GET /health` every 14 minutes, keeping the server alive.

### Option B: UptimeRobot (Free)

1. Go to [uptimerobot.com](https://uptimerobot.com)
2. **Add Monitor** → HTTP(s) → your health URL
3. Set interval to 5 minutes
4. Also doubles as uptime monitoring with email alerts

### Option C: GitHub Actions (Free)

Add `.github/workflows/keepalive.yml`:

```yaml
name: Keep server alive
on:
  schedule:
    - cron: '*/14 * * * *'  # every 14 mins

jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - run: curl -s https://transliteration-engine.onrender.com/health
```

---

## Alternative Platforms

### Railway

- **Pros:** Simpler UI, $5 free credit/month, faster cold starts
- **Deploy:** `railway init` → `railway up`
- **URL:** `https://your-app.up.railway.app`

### Fly.io

- **Pros:** Global edge deployment, 3 free VMs
- **Deploy:** `flyctl launch` → `flyctl deploy`
- **URL:** `https://your-app.fly.dev`

### Hugging Face Spaces

- **Pros:** Free GPU!, ML-focused, great for demos
- **Deploy:** Push to a HF Space repo with `Dockerfile`
- **URL:** `https://your-user-translit.hf.space`
- **Con:** Spaces sleep after 48h of inactivity (even with cron)

---

## Platform Comparison

| Platform | Free Tier | Cold Start | GPU | Best For |
|----------|-----------|------------|-----|----------|
| **Render** | ✅ 512MB RAM | ~30s | ❌ | Production demos |
| **Railway** | $5/month credit | ~15s | ❌ | Fast iteration |
| **Fly.io** | 3 free VMs | ~10s | ❌ | Global low-latency |
| **HF Spaces** | ✅ Free GPU | ~60s | ✅ | ML showcases |
| **localhost** | N/A | 0s | ✅ | Development |

