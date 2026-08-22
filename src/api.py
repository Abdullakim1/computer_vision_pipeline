"""CineForge REST API - Professional Video Generation Studio.

Comprehensive API for text-to-video, image-to-video, storyboarding, and video analysis.

Endpoints
---------
GET  /health                 -> service status & backend info
POST /video/generate          -> text-to-video
POST /video/batch            -> batch generation
POST /video/image            -> image-to-video
POST /video/story            -> storyboard film
POST /video/grade            -> apply color grading
GET  /video/lookpresets      -> list color grading presets
GET  /video/themes           -> list procedural themes
GET  /video/motions          -> list camera motions
POST /video/analyze          -> analyze video quality
POST /video/compilation      -> create video compilation

Run:  PYTHONPATH=. uvicorn src.api:app --reload --port 8000
"""

from __future__ import annotations

import base64
import json
import tempfile
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.forge import CineForgeStudio

app = FastAPI(
    title="CineForge Video Generation Studio",
    version="2.0.0",
    description="Advanced generative video synthesis API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

_studio = CineForgeStudio()
_OUTDIR = Path("outputs/api")
_OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================
# Request Models
# ============================================
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Text prompt for video generation")
    negative_prompt: str = Field("", description="Negative prompt to avoid")
    width: int = Field(960, ge=128, le=3840, description="Video width")
    height: int = Field(540, ge=128, le=2160, description="Video height")
    fps: int = Field(24, ge=8, le=60, description="Frames per second")
    duration: float = Field(4.0, ge=2.0, le=20.0, description="Duration in seconds")
    backend: str = Field("cinematic", description="Generation backend")
    seed: int = Field(None, description="Random seed for reproducibility")
    look: str = Field("argo", description="Color grading preset")
    motion: str = Field("orbit", description="Camera motion")
    style: str = Field("cinematic", description="Generation style")
    enhance_prompt: bool = Field(True, description="Enhance prompt with keywords")

    @field_validator('backend')
    @classmethod
    def validate_backend(cls, v):
        if v not in ['cinematic', 'luma', 'seedance', 'kling']:
            raise ValueError("Unsupported backend. Use 'cinematic', 'luma', 'seedance', or 'kling'")
        return v


class BatchGenerateRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, description="List of prompts to generate")
    backend: str = "cinematic"
    n_per_prompt: int = 1
    width: int = 960
    height: int = 540
    fps: int = 24
    duration: float = 4.0
    seed: int = None
    look: str = "argo"


class ImageRequest(BaseModel):
    width: int = Field(960, ge=128, le=3840)
    height: int = Field(540, ge=128, le=2160)
    fps: int = Field(24, ge=8, le=60)
    duration: float = Field(4.0, ge=2.0, le=20.0)
    look: str = "argo"
    interp: int = Field(1, ge=1, le=6, description="Interpolation factor")
    motion: str = Field("orbit", description="Motion type")


class StoryRequest(GenerateRequest):
    beats: int = Field(3, ge=1, le=8, description="Number of storyboard beats")
    fade: int = Field(6, ge=0, le=12, description="Crossfade duration in frames")


class GradeRequest(BaseModel):
    look: str = Field(..., description="Color grading preset")
    intensity: float = Field(1.0, ge=0.5, le=2.0, description="Grade intensity")


class AnalysisRequest(BaseModel):
    video: str = Field(..., description="Path to video file")
    fps: int = Field(24, ge=1, le=60)


class CompilationRequest(BaseModel):
    clips: List[str] = Field(..., min_length=1, description="Paths to video files")
    transition: str = Field("crossfade", description="Transition type")
    duration: float = Field(3.0, ge=1.0, le=10.0, description="Transition duration")


# ============================================
# Utility Endpoints
# ============================================
@app.get("/health")
def health():
    """Service health and backend status."""
    backends = _studio.backends()
    ready_backends = [b for b in backends if b['ready']]
    return {
        "status": "healthy",
        "version": "2.0.0",
        "generations_count": _studio._generation_count,
        "backends": {
            name: {
                "ready": info['ready'],
                "description": info.get('desc', ''),
            } for name, info in backends.items()
        },
        "ready_backends": ready_backends,
        "total_backends": len(backends),
    }


@app.get("/video/lookpresets")
def get_look_presets():
    """Get list of color grading presets."""
    presets = _studio.get_available_looks()
    return {"presets": presets, "count": len(presets)}


@app.get("/video/themes")
def get_themes():
    """Get list of procedural themes."""
    themes = _studio.get_available_themes()
    return {"themes": themes, "count": len(themes)}


@app.get("/video/motions")
def get_motions():
    """Get list of camera motions."""
    motions = _studio.get_available_motions()
    return {"motions": motions, "count": len(motions)}


@app.get("/video/batch", response_model=dict)
def batch_info():
    """Get batch generation capabilities."""
    return {
        "supported": True,
        "max_prompts": 50,
        "max_per_prompt": 10,
        "default_backend": "cinematic",
        "supported_backends": ["cinematic", "luma", "seedance", "kling"],
    }

# ============================================
# Main Generation Endpoints
# ============================================
@app.post("/video/generate")
def generate(req: GenerateRequest):
    """Generate video from text prompt."""
    try:
        clip = _studio.text_to_video(
            prompt=req.prompt,
            backend=req.backend,
            width=req.width,
            height=req.height,
            fps=req.fps,
            duration=req.duration,
            seed=req.seed,
            negative_prompt=req.negative_prompt,
            look=req.look,
            motion=req.motion,
            style=req.style,
            enhance_prompt=req.enhance_prompt,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    
    seed = req.seed or clip.metadata.get('seed', 0)
    out = _OUTDIR / f"gen_{seed}.mp4"
    mp4 = clip.write_video(out)
    gif = clip.to_gif(str(out.with_suffix(".gif")), fps=min(12, clip.T))
    sheet = clip.contact_sheet(cols=4, rows=1)
    sheet_path = out.with_suffix(".png")
    import cv2
    cv2.imwrite(str(sheet_path), np.ascontiguousarray(sheet[:, :, ::-1]))
    
    return {
        "success": True,
        "file": f"/outputs/{Path(mp4).name}",
        "gif": f"/outputs/{Path(gif).name}",
        "contact_sheet": f"/outputs/{Path(sheet_path).name}",
        "duration": float(clip.duration_s),
        "frames": clip.T,
        "resolution": f"{clip.width}x{clip.height}",
        "fps": float(clip.fps),
        "metrics": clip.metadata.get("metrics", {}),
        "backend": clip.backend,
        "theme": clip.metadata.get("theme", "unknown"),
        "seed": seed,
    }


@app.post("/video/batch")
def batch_generate(req: BatchGenerateRequest):
    """Generate multiple videos from multiple prompts."""
    results = []
    
    for i, prompt in enumerate(req.prompts):
        try:
            clip = _studio.text_to_video(
                prompt=prompt,
                backend=req.backend,
                width=req.width,
                height=req.height,
                fps=req.fps,
                duration=req.duration,
                seed=req.seed,
                look=req.look,
            )
            
            seed = req.seed or clip.metadata.get('seed', i)
            out = _OUTDIR / f"batch_{i}_{seed}.mp4"
            clip.write_video(out)
            
            results.append({
                "index": i,
                "success": True,
                "prompt": prompt,
                "file": f"/outputs/{Path(out).name}",
                "duration": float(clip.duration_s),
                "frames": clip.T,
                "quality": clip.metadata.get("metrics", {}).get("perceptual_quality", 0),
            })
        except RuntimeError as e:
            results.append({
                "index": i,
                "success": False,
                "prompt": prompt,
                "error": str(e),
            })
    
    return {
        "success": True,
        "total": len(req.prompts),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results,
    }


@app.post("/video/image")
def image_generate(req: ImageRequest, image: bytes = Body(..., description="Image file as bytes")):
    """Generate video from static image with Ken Burns effect."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(image)
    tmp.close()
    
    try:
        clip = _studio.image_to_video(
            tmp.name,
            width=req.width,
            height=req.height,
            fps=req.fps,
            duration=req.duration,
            look=req.look,
            interp=req.interp,
            motion=req.motion,
        )
    finally:
        Path(tmp.name).unlink()
    
    out = _OUTDIR / "image_to_video.mp4"
    clip.write_video(out)
    
    return {
        "success": True,
        "file": f"/outputs/{out.name}",
        "duration": float(clip.duration_s),
        "frames": clip.T,
        "metrics": clip.metadata.get("metrics", {}),
    }


@app.post("/video/story")
def story(req: StoryRequest):
    """Generate a storyboard short film."""
    try:
        clip = _studio.direct(
            prompt=req.prompt,
            beats=req.beats,
            backend=req.backend,
            width=req.width,
            height=req.height,
            fps=req.fps,
            seed=req.seed,
            look=req.look,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    
    out = _OUTDIR / f"story_{req.beats}_beats.mp4"
    clip.write_video(out)
    
    return {
        "success": True,
        "file": f"/outputs/{out.name}",
        "beats": req.beats,
        "duration": float(clip.duration_s),
        "frames": clip.T,
        "metrics": clip.metadata.get("metrics", {}),
        "shots": clip.metadata.get("shots", []),
    }


@app.post("/video/grade")
def video_grade(file: UploadFile = File(...), look: str = Query(...), intensity: float = Query(1.0)):
    """Apply color grading to uploaded video."""
    import cv2
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    
    try:
        clip = _studio.text_to_video(tmp_path, look=look)
        clip = _studio.regrade(clip, look)
        
        out = _OUTDIR / f"graded_{look}_{file.filename}.mp4"
        clip.write_video(out)
        
        return {
            "success": True,
            "file": f"/outputs/{out.name}",
            "graded_look": look,
            "intensity": intensity,
        }
    finally:
        Path(tmp_path).unlink()


@app.post("/video/analyze")
def analyze_video(req: AnalysisRequest):
    """Analyze video quality with comprehensive metrics."""
    import cv2
    
    cap = cv2.VideoCapture(req.video)
    frames = []
    while len(frames) < 300 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    if not frames:
        raise HTTPException(status_code=400, detail="No frames extracted from video")
    
    clip = _studio.text_to_video("analysis", backend="cinematic", frames=frames, fps=req.fps)
    metrics_data = clip.metadata.get("metrics", {})
    
    return {
        "success": True,
        "video": req.video,
        "frames": len(frames),
        "metrics": metrics_data,
    }


@app.post("/video/compilation")
def create_compilation(req: CompilationRequest):
    """Create a compilation video from multiple clips."""
    try:
        clips = []
        for clip_path in req.clips:
            clip = _studio.text_to_video(clip_path, backend="cinematic")
            clips.append(clip)
        
        compilation = _studio.create_compilation(
            clips=clips,
            transition=req.transition,
            duration=req.duration,
        )
        
        out = _OUTDIR / f"compilation_{req.transition}.mp4"
        compilation.write_video(out)
        
        return {
            "success": True,
            "file": f"/outputs/{out.name}",
            "clips_used": len(req.clips),
            "transition": req.transition,
            "transition_duration": req.duration,
            "total_duration": float(compilation.duration_s),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# File Serving
# ============================================
@app.get("/outputs/{filename}")
def get_output(filename: str):
    """Serve generated output files."""
    file_path = _OUTDIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    content_type = "video/mp4" if filename.endswith(".mp4") else "image/png"
    return FileResponse(file_path, media_type=content_type)
