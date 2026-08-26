# 🎬 CineForge - Professional Video Generation Studio

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Version](https://img.shields.io/badge/Version-2.0.0-orange)

[English](#-cineforge)

**Production-Grade Video Generation Platform for AI/ML Professionals**

</div>

---

## 📖 Overview

CineForge is an advanced, production-ready video generation studio that bridges the gap between AI-driven content creation and traditional cinematic workflows. Built with a modular architecture, it supports:

- **Procedural Video Generation** - GPU-free generative engine with cinematic quality
- **Cloud API Integration** - Seamless integration with Seedance 2.0 & Kling 3.0
- **Advanced Motion** - 25+ camera motions with optical flow interpolation
- **Professional Grading** - 18+ color grading presets
- **Comprehensive Metrics** - Quality analysis for production pipelines
- **One-Command Portfolio** - A self-contained HTML showcase gallery

---

## ✨ Key Features

### 🎥 Procedural Generation
- **14 Procedural Themes**: Aurora, Sunset, Golden Hour, Neon City, Ocean, Mountains, Mono, Campfire, Cyber, Moody, Meadow, Storm, City, Apocalypse
- **Prompt → Theme auto-resolution**: free text is mapped to the closest theme, so a request like *"zombies walking on the street in a city"* renders an apocalyptic urban scene instead of a random aurora
- **Urban Scene Engine**: night city skyline with lit windows, a street, and a crowd of **walking zombie silhouettes** (6–12 animated figures)
- **25+ Camera Motions**: Orbit, Pan, Zoom, Dolly, Crane, Handheld, Dutch Tilt, Jib, Tracking, and more
- **6 Particle Systems**: Fireflies, Embers, Snow, Rain, Foam, Data streams
- **Atmospheric Effects**: Volumetric glow, lens flares, fog, haze, horizon glow

### 🌐 Cloud Integration
- **Colab Backend (recommended, photoreal)**: [Wan](https://huggingface.co/Wan-AI) **text-to-video AND image-to-video** hosted on your **Colab GPU** — an A100 (Colab Pro) gives the most powerful output; it also falls back to a free T4. See [colab/CineForge_Colab_Video_Server.ipynb](colab/CineForge_Colab_Video_Server.ipynb)
- **Kaggle Backend**: the same Wan server stack hosted on a **Kaggle GPU** (free T4/P100 works) behind a cloudflared tunnel. Run a notebook from [kaggle/](kaggle/), paste the printed `trycloudflare.com` URL into `.env` as `KAGGLE_BASE_URL`, then pick the **kaggle** backend. Text-to-video only on free-tier GPUs; image-to-video needs a ≥30 GB GPU.
- **Luma Backend**: Luma AI Dream Machine API (Ray 2) for photorealistic text-to-video
- **Seedance Backend**: ByteDance Provenance integration with style presets
- **Kling Backend**: Kua'you Video 3.0 API with extended duration support
- **Automatic Polling**: Background task management for cloud generation

#### Realistic video generation (via Colab)
The default `cinematic` backend is procedural, so it renders stylized/animated scenes. For
**photorealistic** text-to-video *and* image-to-video:

1. Open `colab/CineForge_Colab_Video_Server.ipynb` in Google Colab, set Runtime → **GPU** (A100 on Colab Pro), run all cells.
2. Copy the printed `PUBLIC API URL` into `.env`: `COLAB_BASE_URL=<url>`
3. Open the studio UI and pick the **colab** backend (or via CLI):
   ```bash
   python -m src.main --prompt "zombies walking on the street in a city" --backend colab
   ```
   Prompts are auto-enhanced with live-action realism keywords; pass `style="raw"` in `extras` to disable. First run downloads model weights to Colab; the URL expires when the runtime disconnects — re-run the last cells and update `.env`.

### 🎨 Cinematic Grading
- **18+ Color Presets**: Argo, Teal-Orange, Golden, Noir, Cyber, Vintage, and more
- **Advanced Effects**: Bokeh, Chromatic Aberration, Scanlines, Film Grain
- **Custom Grading**: Per-frame adjustment with intensity control

### 📊 Quality Metrics
- **Motion Analysis**: Optical flow, temporal diversity, motion continuity
- **Visual Quality**: Sharpness, colorfulness, noise level, dynamic range
- **Perceptual Scoring**: AI-powered quality assessment

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cineforge.git
cd cineforge

# Install dependencies
pip install -r requirements.txt

# Run the demo
python -m src.main demo
```

### Using the CLI

```bash
# List available backends and themes
cineforge info

# Generate a video from text
python -m src.main gen "cinematic aurora over mountains" --backend cinematic --motion orbit

# Free-text prompts auto-resolve to a theme — zombies produce an urban scene
python -m src.main gen "zombies walking on the street in a city at night" --out outputs/zombies.mp4

# Generate a storyboard film
python -m src.main story "A lone traveler crosses a dune" --beats 5

# Generate a showcase portfolio
python -m src.main showcase --outdir media/showcase

# Build a self-contained HTML portfolio gallery (recommended)
python -m src.main portfolio --outdir media/portfolio

# View system statistics
cineforge stats
```

### Using the Web Interface

```bash
# Start the API (serves the studio web UI at the root URL)
cd /home/kim/computer_vision_pipeline
PYTHONPATH=. uvicorn src.api:app --port 8000

# Then open http://localhost:8000 in your browser
```

### Using the REST API

```bash
# Start the API server
PYTHONPATH=. uvicorn src.api:app --reload --port 8000

# Test the API
curl http://localhost:8000/health

# Generate a video via API
curl -X POST "http://localhost:8000/video/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "aurora over mountains", "backend": "cinematic", "duration": 4.0}'
```

---

## 🖼️ Portfolio & Showcase

One command turns the whole engine into a **shareable, self-contained portfolio** —
a deliverable you can open in a browser, drop into a presentation, or share as a
single `index.html` file with every preview embedded inline:

```bash
# Build the full 12-theme portfolio (clips, gifs, posters, report, gallery)
python -m src.main portfolio --outdir media/portfolio

# Build a curated subset
python -m src.main portfolio --themes aurora cyber neon --outdir media/portfolio
```

What it produces in `media/portfolio/`:

| Artifact | Contents |
|----------|----------|
| `index.html` | Branded, **self-contained** gallery — base64 previews embedded, opens anywhere |
| `<theme>.mp4`  | Full-resolution clip for each theme |
| `<theme>.gif`  | Lightweight animated preview (GIF) |
| `<theme>.jpg`  | Representative poster frame |
| `metrics_report.txt` | Per-clip perceptual quality, sharpness & colorfulness |

Each clip pairs a theme with a distinct camera move (orbit, pan, dolly, crane,
tilt, handheld…) so the portfolio demonstrates the full range of the motion
engine at a glance. You can launch the same workflow from the **Portfolio** tab
of the web UI.

---

## 🏗️ System Architecture

```
CineForge Studio
├── Procedural Backend (GPU-free)
│   ├── Theme Engine (14 themes)
│   ├── Particle System
│   ├── Atmospheric Effects
│   └── Camera Motion Controller
├── Cloud Backend Adapter
│   ├── Seedance Integration
│   ├── Kling Integration
│   └── Task Management
├── Effects Pipeline
│   ├── Color Grading
│   ├── Advanced Effects
│   └── Post-Processing
├── Metrics Engine
│   ├── Optical Flow Analysis
│   ├── Quality Assessment
│   └── Motion Analysis
└── Orchestration Layer
    ├── CLI Interface
    ├── Web UI (Gradio)
    └── REST API (FastAPI)
```

---

## 🎨 Procedural Themes

| Theme | Palette | Look | Particles |
|-------|---------|------|-----------|
| Aurora | Polar Night | Cyber | Fireflies |
| Sunset | Dusk Sky | Golden | Embers |
| Golden Hour | Morning Light | Golden | Dust |
| Neon City | Night Lights | Cyber | Lights |
| Ocean Waves | Blue Depths | Argo | Foam |
| Mountain Range | Distant Peaks | Vintage | Mist |
| Monochrome | B&W Film | Noir | Snow |
| Campfire | Night Glow | Teal-Orange | Embers |
| Cyber Future | Hologram | Cyber | Data |
| Moody Fog | Dark Tone | Noir | Mist |
| Meadow | Grass Field | Golden | Pollen |
| Storm | Lightning | Argo | Rain |
| City | Urban Skyline | Cyber | Lights |
| Apocalypse | Ruins & Zombies | Moody | Ash |

---

## 🎬 Camera Motions

- **Orbit**: Gentle circular motion
- **Pan Left/Right**: Horizontal movement
- **Zoom In/Out**: Focus manipulation
- **Dolly In/Out**: Forward/backward movement
- **Crane Up/Down**: Vertical elevation
- **Handheld**: Natural camera shake
- **Dutch Tilt**: Off-kilter dramatic frame
- **Jib**: Rising sweeping arc shot
- **Tracking**: Steady lateral follow
- **Static**: Locked-off tripod frame

---

## 🖥️ Interfaces

### CLI

```bash
python -m src.main info                  # backend & environment status
python -m src.main gen "aurora over a valley" --backend cinematic
python -m src.main story "a sun sets over ancient ruins" --beats 3
python -m src.main i2v photo.jpg
python -m src.main grade clip.mp4 --look golden
python -m src.main metrics clip.mp4
python -m src.main portfolio --outdir media/portfolio
```

### Web UI (recommended)

A modern single-page studio served directly by the API:

```bash
PYTHONPATH=. uvicorn src.api:app --port 8000   # open http://localhost:8000
```

Features a dark cinematic interface with three panels:

| Panel | What it does |
|-------|--------------|
| **Text → Video** | Prompt + negative prompt, backend picker, duration/FPS/resolution/steps/CFG/seed |
| **Image → Video** | Drag-&-drop a still image + optional motion prompt; runs real diffusion I2V (Colab Wan / local / Kling / Seedance) |
| **Grade / Analyze** | Re-color an existing clip with 18+ looks and inspect quality metrics |

Live backend health is shown in the header; the colab backend auto-reports readiness
from `COLAB_BASE_URL`.

### REST API (FastAPI)

```bash
PYTHONPATH=. uvicorn src.api:app --reload --port 8000   # docs at /docs
```

---

## 📄 License

MIT
