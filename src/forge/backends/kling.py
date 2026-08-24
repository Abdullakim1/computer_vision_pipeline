"""Klong (Kua'you) HTTP adapter - Kling Video 3.0 API.

Integration point for the Klong video-3.0 API (image-to-video, motion,
effects). Reads ``KLING_API_KEY`` from env; otherwise the adapter self-reports
unavailable. This mirrors how studios swap remote vendors behind one facade.

This backend implements:
- Text-to-video generation with extended duration support
- Image-to-video with motion presets
- Camera direction and zoom control
- Animation style presets
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
class KlingRequest:
    """Request payload structure for Kling API."""
    prompt: str
    negative_prompt: str = "blurry, low quality, worst quality, jpeg artifacts"
    image: Optional[str] = None
    motion: str = "camera_orbit"
    motion_strength: float = 0.5
    camera_direction: str = "isometric"
    zoom_direction: str = "center"
    duration: int = 4
    fps: int = 24
    seed: Optional[int] = None
    style: str = "cinematic"
    aspect_ratio: str = "16:9"
    safety_check: bool = True


@dataclass
class KlingResponse:
    """Response payload structure from Kling API."""
    task_id: str
    status: str
    url: Optional[str] = None
    preview_url: Optional[str] = None
    frames: Optional[int] = None
    duration: Optional[float] = None
    error: Optional[str] = None


class KlingBackend(GeneratorBackend):
    name = "kling"

    # Animation style presets
    STYLE_PRESETS = {
        "cinematic": "realistic",
        "anime": "anime",
        "3d": "3d_render",
        "realistic": "photorealistic",
        "artistic": "digital_art",
        "animated": "animation",
        "moody": "noir",
        "colorful": "vibrant",
    }

    # Camera motion presets
    MOTION_PRESETS = {
        "camera_orbit": "center",
        "pan_left": "left",
        "pan_right": "right",
        "zoom_in": "zoom_in",
        "zoom_out": "zoom_out",
        "dolly_in": "zoom_in",
        "dolly_out": "zoom_out",
        "tracking": "tracking",
        "handheld": "handheld",
        "crane_up": "crane",
        "crane_down": "crane",
        "tilt_up": "tilt_up",
        "tilt_down": "tilt_down",
    }

    # Zoom direction presets
    ZOOM_PRESETS = {
        "center": "center",
        "top_left": "top_left",
        "top_right": "top_right",
        "bottom_left": "bottom_left",
        "bottom_right": "bottom_right",
    }

    def __init__(self, **env):
        super().__init__(**env)
        self.api_key = os.getenv("KLING_API_KEY", "")
        self.base_url = (os.getenv("KLING_BASE_URL")
                          or os.getenv("KLING_API_ENDPOINT")
                          or "https://api.klingai.com")
        self.api_version = os.getenv("KLING_API_VERSION", "v1")
        self.timeout = int(os.getenv("KLING_TIMEOUT", "300"))
        self.poll_interval = int(os.getenv("KLING_POLL_INTERVAL", "5"))

    def check(self) -> bool:
        """Check if Kling backend is available."""
        return bool(self.api_key)

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64 for API upload."""
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            return image_data
        except Exception:
            return None

    def _enhance_prompt(self, prompt: str, style: str) -> str:
        """Enhance prompt with style and technical details."""
        style = self.STYLE_PRESETS.get(style, "cinematic")
        enhancements = [
            "4K UHD",
            "professional grade",
            "highly detailed",
            "smooth motion",
            "cinematic composition",
        ]

        enhanced = prompt
        for enh in enhancements:
            enhanced += f", {enh}"

        if style in ["cinematic", "realistic"]:
            enhanced += ", photorealistic, 8K, film grain"
        elif style == "anime":
            enhanced += ", anime style, vibrant, Studio Ghibli"
        elif style == "3d":
            enhanced += ", 3D rendered, Octane render, ray tracing"

        return enhanced.strip()

    def _format_response(self, data: Dict[str, Any]) -> KlingResponse:
        """Parse API response into our dataclass."""
        return KlingResponse(
            task_id=data.get("task_id", ""),
            status=data.get("status", "unknown"),
            url=data.get("url"),
            preview_url=data.get("preview_url"),
            frames=data.get("frames"),
            duration=data.get("duration"),
            error=data.get("error"),
        )

    def _poll_task(self, task_id: str) -> Dict[str, Any]:
        """Poll Kling API for task completion."""
        url = f"{self.base_url}/{self.api_version}/video/tasks/{task_id}"

        for attempt in range(self.timeout // self.poll_interval):
            try:
                response = requests.get(url, timeout=self.poll_interval)
                data = response.json()

                status = data.get("status")
                if status in ["completed", "succeeded", "finished", "success"]:
                    return data
                elif status in ["failed", "error", "error_occurred"]:
                    raise RuntimeError(f"Task failed: {data.get('error', 'Unknown error')}")
                elif status in ["pending", "processing", "generating", "queueing"]:
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
        """Generate video using Kling API."""
        if not self.api_key:
            raise RuntimeError(
                "Kling adapter needs KLING_API_KEY; configure in .env "
                "or use 'cinematic' backend for offline synthesis."
            )

        motion_preset = req.extras.get("motion", req.extras.get("camera", "camera_orbit"))
        motion_preset = self.MOTION_PRESETS.get(motion_preset, "camera_orbit")

        zoom_preset = req.extras.get("zoom", req.extras.get("zoom_direction", "center"))
        zoom_preset = self.ZOOM_PRESETS.get(zoom_preset, "center")

        style = req.extras.get("style", "cinematic")
        style = self.STYLE_PRESETS.get(style, "cinematic")

        kling_req = KlingRequest(
            prompt=self._enhance_prompt(req.prompt, style),
            negative_prompt=req.negative_prompt,
            image=self._encode_image(req.first_frame) if req.first_frame else None,
            motion=motion_preset,
            motion_strength=req.extras.get("motion_strength", 0.5),
            camera_direction=zoom_preset,
            zoom_direction=zoom_preset,
            duration=min(max(req.duration, 2), 16),
            fps=min(max(req.fps, 8), 60),
            seed=req.seed or int(uuid.uuid4().int % (1 << 32)),
            style=style,
            aspect_ratio=f"{req.width}:{req.height}",
            safety_check=req.extras.get("safety_check", True),
        )

        task_id = self._submit_task(kling_req)
        task_data = self._poll_task(task_id)

        if task_data.get("status") not in ["completed", "succeeded"]:
            error_msg = task_data.get("error", "Unknown error")
            raise RuntimeError(f"Kling generation failed: {error_msg}")

        video_url = task_data.get("url") or task_data.get("video_url")
        if not video_url:
            raise RuntimeError("No video URL returned from Kling API")

        temp_dir = os.path.join("outputs", "kling", "temp")
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
                "zoom": zoom_preset,
                "frames_extracted": len(frames),
                "api_response": task_data,
            }
        )

        return clip

    def _submit_task(self, req: KlingRequest) -> str:
        """Submit generation task to Kling API."""
        url = f"{self.base_url}/{self.api_version}/video/generate"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = asdict(req)

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
