"""Seedance (ByteDance) HTTP adapter - Provenance Video Generation API.

Calls a Seedance-style provided API for text/image-to-video. Auto-detects an
API key to decide readiness. Provide credentials via env (``SEEDANCE_API_KEY``,
``SEEDANCE_BASE_URL``). When absent the adapter degrades cleanly.

This backend implements:
- Text-to-video generation with style control
- Image-to-video (text-guided motion)
- Camera motion presets
- Style presets (cinematic, anime, 3D render, etc.)
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import requests

from ..types import GeneratedClip, GenerationRequest
from .base import GeneratorBackend


@dataclass
class SeedanceRequest:
    """Request payload structure for Seedance API."""
    prompt: str
    negative_prompt: str = "blurry, low quality, worst quality, jpeg artifacts"
    image: Optional[str] = None
    motion: str = "camera_orbit"
    motion_strength: float = 0.7
    camera_angle: str = "isometric"
    aspect_ratio: str = "16:9"
    duration: int = 4
    fps: int = 24
    seed: Optional[int] = None
    style: str = "cinematic"
    enhance_prompt: bool = True


@dataclass
class SeedanceResponse:
    """Response payload structure from Seedance API."""
    task_id: str
    status: str
    url: Optional[str] = None
    preview_url: Optional[str] = None
    frames: Optional[int] = None
    duration: Optional[float] = None
    error: Optional[str] = None


class SeedanceBackend(GeneratorBackend):
    name = "seedance"

    STYLE_PRESETS = {
        "cinematic": "photorealistic_3d",
        "anime": "anime_style",
        "3d": "3d_render",
        "realistic": "photorealistic",
        "artistic": "digital_art",
        "moody": "film_noir",
        "vibrant": "saturated_color",
    }

    MOTION_PRESETS = {
        "camera_orbit": "orbit",
        "pan_left": "pan_left",
        "pan_right": "pan_right",
        "zoom_in": "zoom_in",
        "zoom_out": "zoom_out",
        "dolly_in": "dolly_in",
        "dolly_out": "dolly_out",
        "tracking": "tracking",
        "handheld": "handheld_shake",
        "tilt": "tilt_up",
        "crane": "crane_shot",
    }

    def __init__(self, **env):
        super().__init__(**env)
        self.api_key = os.getenv("SEEDANCE_API_KEY", "")
        # Seedance is ByteDance's video model served by Volcano Engine (Ark).
        # The endpoint is pluggable via env so the adapter can target any
        # Seedance-compatible gateway. (Do not confuse with Kling.)
        self.base_url = (os.getenv("SEEDANCE_BASE_URL")
                          or os.getenv("SEEDANCE_API_ENDPOINT")
                          or "https://ark.cn-beijing.volces.com/api")
        self.api_version = os.getenv("SEEDANCE_API_VERSION", "v3")
        self.timeout = int(os.getenv("SEEDANCE_TIMEOUT", "300"))
        self.poll_interval = int(os.getenv("SEEDANCE_POLL_INTERVAL", "3"))

    def check(self) -> bool:
        """Check if Seedance backend is available."""
        return bool(self.api_key)

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64 for API upload."""
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{image_data}"
        except Exception:
            return None

    def _enhance_prompt(self, prompt: str, style: str) -> str:
        """Enhance prompt with style and technical details."""
        style = self.STYLE_PRESETS.get(style, "cinematic")
        enhancements = [
            "4K high quality",
            "professional lighting",
            "depth of field",
            "motion blur",
            "atmospheric",
        ]

        enhanced = prompt
        for enh in enhancements:
            enhanced += f", {enh}"

        if style in ["cinematic", "realistic"]:
            enhanced += ", ultra detailed, 8K, unreal engine 5 render"
        elif style == "anime":
            enhanced += ", anime style, vibrant colors, Studio Ghibli inspired"
        elif style == "3d":
            enhanced += ", 3D render, Octane render, ray tracing"

        return enhanced.strip()

    def _format_response(self, data: Dict[str, Any]) -> SeedanceResponse:
        """Parse API response into our dataclass."""
        return SeedanceResponse(
            task_id=data.get("task_id", ""),
            status=data.get("status", "unknown"),
            url=data.get("url"),
            preview_url=data.get("preview_url"),
            frames=data.get("frames"),
            duration=data.get("duration"),
            error=data.get("error"),
        )

    def _poll_task(self, task_id: str) -> Dict[str, Any]:
        """Poll Seedance API for task completion."""
        url = f"{self.base_url}/{self.api_version}/video/tasks/{task_id}"

        for attempt in range(self.timeout // self.poll_interval):
            try:
                response = requests.get(url, timeout=self.poll_interval)
                data = response.json()

                status = data.get("status")
                if status in ["completed", "succeeded", "finished"]:
                    return data
                elif status in ["failed", "failed"]:
                    raise RuntimeError(f"Task failed: {data.get('error', 'Unknown error')}")
                elif status in ["pending", "processing"]:
                    pass
                else:
                    break

            except requests.exceptions.Timeout:
                pass
            except Exception as e:
                raise RuntimeError(f"Error polling task: {e}")

            import time
            time.sleep(self.poll_interval)

        raise RuntimeError("Task polling timeout - generation incomplete")

    def _download_frames(self, video_url: str, output_dir: str, frame_count: int) -> List[np.ndarray]:
        """Download generated video and extract frames."""
        try:
            response = requests.get(video_url, timeout=60)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to download video: {response.status_code}")

            temp_path = os.path.join(output_dir, f"temp_{uuid.uuid4()}.mp4")
            with open(temp_path, "wb") as f:
                f.write(response.content)

            cap = cv2.VideoCapture(temp_path)
            frames = []

            while len(frames) < frame_count and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            cap.release()
            os.unlink(temp_path)

            return frames

        except Exception as e:
            raise RuntimeError(f"Failed to download/extract frames: {e}")

    def generate(self, req: GenerationRequest) -> GeneratedClip:
        """Generate video using Seedance API."""
        if not self.api_key:
            raise RuntimeError(
                "Seedance adapter needs SEEDANCE_API_KEY; configure in .env "
                "or use 'cinematic' backend for offline synthesis."
            )

        motion_preset = req.extras.get("motion", req.extras.get("camera", "camera_orbit"))
        motion_preset = self.MOTION_PRESETS.get(motion_preset, "camera_orbit")

        style = req.extras.get("style", "cinematic")
        style = self.STYLE_PRESETS.get(style, "cinematic")

        seedance_req = SeedanceRequest(
            prompt=self._enhance_prompt(req.prompt, style),
            negative_prompt=req.negative_prompt,
            image=self._encode_image(req.first_frame) if req.first_frame else None,
            motion=motion_preset,
            motion_strength=req.extras.get("motion_strength", 0.7),
            camera_angle=req.extras.get("camera_angle", "isometric"),
            aspect_ratio=f"{req.width}:{req.height}",
            duration=min(max(req.duration, 2), 10),
            fps=min(max(req.fps, 8), 60),
            seed=req.seed or int(uuid.uuid4().int % (1 << 32)),
            style=style,
            enhance_prompt=req.extras.get("enhance_prompt", True),
        )

        task_id = self._submit_task(seedance_req)
        task_data = self._poll_task(task_id)

        if task_data.get("status") not in ["completed", "succeeded"]:
            error_msg = task_data.get("error", "Unknown error")
            raise RuntimeError(f"Seedance generation failed: {error_msg}")

        video_url = task_data.get("url") or task_data.get("video_url")
        if not video_url:
            raise RuntimeError("No video URL returned from Seedance API")

        temp_dir = os.path.join("outputs", "seedance", "temp")
        os.makedirs(temp_dir, exist_ok=True)

        frame_count = req.n_frames
        frames = self._download_frames(video_url, temp_dir, frame_count)

        if not frames:
            raise RuntimeError("No frames extracted from generated video")

        clip = GeneratedClip(
            prompt=req.prompt,
            backend=self.name,
            frames=np.stack(frames),
            fps=float(req.fps),
            metadata={
                "task_id": task_id,
                "style": style,
                "motion": motion_preset,
                "frames_extracted": len(frames),
                "api_response": task_data,
            }
        )

        return clip

    def _submit_task(self, req: SeedanceRequest) -> str:
        """Submit generation task to Seedance API."""
        url = f"{self.base_url}/{self.api_version}/video/generate"

        payload = asdict(req)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=60,
                headers=headers,
            )

            if response.status_code != 200:
                error_text = response.text
                try:
                    error_data = response.json()
                    error_text = error_data.get("error", error_text)
                except:
                    pass
                raise RuntimeError(f"API request failed: {response.status_code} - {error_text}")

            result = response.json()

            if "task_id" not in result:
                raise RuntimeError(f"API response missing task_id: {result}")

            return result["task_id"]

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error submitting task: {e}")
