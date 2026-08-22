"""Luma AI Dream Machine backend - Luma Labs Video Generation API.

Integration point for the Luma AI Dream Machine API (text-to-video,
image-to-video).  Reads LUMA_API_KEY from env; otherwise the adapter
self-reports unavailable.

This backend implements:
- Text-to-video generation with the Ray 2 model
- Image-to-video (via public image URL in keyframes)
- Camera motion control via prompt language
- Loop support
- Extended duration support (2-10 s)

API reference: https://docs.lumalabs.ai/
"""

from __future__ import annotations

import base64
import json
import os
import uuid
import time
import math
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import requests

from ..types import GeneratedClip, GenerationRequest
from .base import GeneratorBackend


@dataclass
class LumaRequest:
    """Request payload structure for Luma Dream Machine API."""
    prompt: str
    model: str = "ray-2"
    resolution: Optional[str] = None       # e.g. "720p", "1080p", "4k"
    aspect_ratio: str = "16:9"
    duration: int = 5                       # seconds; Luma formats as "5s"
    loop: bool = False
    image: Optional[str] = None             # Public URL for image-to-video
    negative_prompt: str = "blur, low quality, worst quality, jpeg artifacts"
    motion: str = "camera_orbit"
    seed: Optional[int] = None
    style_preset: Optional[str] = None


@dataclass
class LumaResponse:
    """Response payload structure from Luma Dream Machine API."""
    id: str
    state: str                              # "queued", "dreaming", "completed", "failed"
    assets: Optional[Dict[str, str]] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    failure_reason: Optional[str] = None


class LumaBackend(GeneratorBackend):
    """Luma Dream Machine cloud backend (text-to-video / image-to-video)."""

    name = "luma"

    # Camera motions — Luma Dream Machine controls the virtual camera
    # through language embedded in the prompt.  These descriptions are
    # appended to the user's prompt to direct the camera.
    MOTION_PRESETS = {
        "camera_orbit": "dynamic 360-degree camera orbit around the subject",
        "pan_left": "slow camera pan to the left",
        "pan_right": "slow camera pan to the right",
        "zoom_in": "smooth dolly zoom in toward the subject",
        "zoom_out": "smooth dolly zoom out from the subject",
        "dolly_in": "dolly forward toward the subject",
        "dolly_out": "dolly backward away from the subject",
        "tracking": "moving tracking shot following the action",
        "crane_up": "crane shot rising from low to high angle",
        "crane_down": "crane shot descending from high to low angle",
        "handheld": "handheld camera movement with subtle shake",
        "tilt_up": "camera tilts upward",
        "tilt_down": "camera tilts downward",
        "static": "static camera, no movement",
    }

    # Aspect ratios supported by Luma Dream Machine
    ASPECT_RATIOS = {"16:9", "9:16", "1:1", "4:3", "21:9"}

    # Motion strengths (informational; Luma uses prompt language, not a float)
    MOTION_STRENGTHS = {
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "extreme": 0.9,
    }

    # Resolution strings accepted by the Dream Machine API
    _RESOLUTION_MAP = {540: "540p", 720: "720p", 1080: "1080p", 2160: "4k"}

    def __init__(self, **env):
        super().__init__(**env)
        self.api_key = os.getenv("LUMA_API_KEY", "")
        self.base_url = os.getenv("LUMA_BASE_URL", "https://api.lumalabs.ai/dream-machine/v1")
        self.api_version = os.getenv("LUMA_API_VERSION", "v1")
        self.timeout = int(os.getenv("LUMA_TIMEOUT", "300"))
        self.poll_interval = int(os.getenv("LUMA_POLL_INTERVAL", "5"))

    def check(self) -> bool:
        """Check if Luma backend is available."""
        return bool(self.api_key)

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64 for local storage / fallback use.

        Note: the Luma API does **not** accept base64 images directly —
        it requires a publicly accessible URL.  Use ``_save_first_frame``
        and provide the URL via ``extras['image_url']`` for I2V.
        """
        try:
            with open(image_path, "rb") as f:
                img_data = f.read()
            return base64.b64encode(img_data).decode("utf-8")
        except Exception as e:
            print(f"Warning: Could not encode image: {e}")
            return None

    def _save_first_frame(self, first_frame, temp_dir: str) -> Optional[str]:
        """Save a numpy array (or path) to disk and return the file path."""
        if first_frame is None:
            return None
        os.makedirs(temp_dir, exist_ok=True)
        path = os.path.join(temp_dir, f"first_frame_{uuid.uuid4().hex[:8]}.jpg")
        if isinstance(first_frame, str):
            return first_frame if os.path.isfile(first_frame) else None
        # Assume numpy array (H, W, 3) RGB uint8
        bgr = first_frame[:, :, ::-1]  # RGB -> BGR
        cv2.imwrite(path, bgr)
        return path

    def _enhance_prompt(self, prompt: str, style: str = None) -> str:
        """Enhance prompt with style keywords."""
        enhancements = {
            "cinematic": "cinematic, photorealistic, 8k, highly detailed",
            "anime": "anime style, 2D animation",
            "3d": "3D render, Pixar-style, CGI",
            "realistic": "photorealistic, 8K, highly detailed",
            "moody": "moody atmosphere, cinematic lighting",
        }
        enhancement = enhancements.get(style, "")
        return f"{prompt}, {enhancement}" if enhancement else prompt

    def _resolution_for(self, width: int, height: int) -> Optional[str]:
        """Map pixel height to a Luma resolution string."""
        return self._RESOLUTION_MAP.get(height)

    @staticmethod
    def _aspect_ratio_for(width: int, height: int) -> str:
        """Compute a simplified aspect-ratio string from dimensions."""
        from math import gcd
        g = gcd(width, height)
        return f"{width // g}:{height // g}"

    def _build_prompt(self, prompt: str, style: str, motion: str) -> str:
        """Enhance the prompt with style keywords and camera-motion language."""
        enhanced = self._enhance_prompt(prompt, style)
        motion_desc = self.MOTION_PRESETS.get(motion, self.MOTION_PRESETS["camera_orbit"])
        return f"{enhanced}, {motion_desc}"

    def _submit_task(self, req: LumaRequest) -> str:
        """Submit generation task to Luma Dream Machine API."""
        url = f"{self.base_url}/generations"

        payload: Dict[str, Any] = {
            "prompt": req.prompt,
            "model": req.model,
        }

        if req.resolution:
            payload["resolution"] = req.resolution

        if req.duration:
            payload["duration"] = f"{req.duration}s"

        if req.aspect_ratio:
            payload["aspect_ratio"] = req.aspect_ratio

        if req.loop:
            payload["loop"] = True

        # Image-to-video: Luma requires a publicly accessible image URL
        # passed via the keyframes structure (base64 is not accepted).
        if req.image:
            payload["keyframes"] = {
                "frame0": {"type": "image", "url": req.image}
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                url, json=payload, timeout=60, headers=headers,
            )

            if response.status_code not in (200, 201):
                error_text = response.text
                try:
                    error_data = response.json()
                    error_text = error_data.get("detail", error_data.get("error", error_text))
                except Exception:
                    pass
                raise RuntimeError(
                    f"Luma API request failed: {response.status_code} - {error_text}"
                )

            result = response.json()

            if "id" not in result:
                raise RuntimeError(f"Luma API response missing id: {result}")

            return result["id"]

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error submitting Luma task: {e}")

    def _poll_task(self, task_id: str) -> Dict[str, Any]:
        """Poll task status until completion.

        Luma returns ``state`` with values from the set
        ``queued``, ``dreaming``, ``completed``, ``failed``.
        """
        url = f"{self.base_url}/generations/{task_id}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        max_wait = self.timeout
        start_time = time.time()

        while True:
            if time.time() - start_time > max_wait:
                raise RuntimeError(f"Luma task {task_id} timeout after {max_wait}s")

            try:
                response = requests.get(url, timeout=30, headers=headers)
                result = response.json()
                state = result.get("state", "queued")

                if state == "completed":
                    return result
                elif state == "failed":
                    error_msg = (
                        result.get("failure_reason")
                        or result.get("error")
                        or "Unknown error"
                    )
                    raise RuntimeError(f"Luma generation failed: {error_msg}")

                # Still processing (queued / dreaming) — wait and retry
                time.sleep(self.poll_interval)

            except requests.exceptions.RequestException:
                time.sleep(self.poll_interval)
                continue

    def _download_video(self, video_url: str, output_path: str, max_retries: int = 3) -> str:
        """Download video from URL."""
        import shutil

        for attempt in range(max_retries):
            try:
                response = requests.get(video_url, timeout=300, stream=True)

                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        shutil.copyfileobj(response.raw, f)
                    return output_path
                else:
                    response.raise_for_status()

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise RuntimeError(f"Failed to download video after {max_retries} attempts: {e}")

        raise RuntimeError(f"Failed to download video after {max_retries} attempts")

    def _download_frames(self, video_path: str, max_frames: int) -> List[np.ndarray]:
        """Extract frames from downloaded video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"No frames found in video: {video_path}")

        # Sample evenly up to max_frames; if the video is shorter,
        # repeat the last frame to avoid an empty array.
        if total_frames >= max_frames:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        else:
            frame_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
            pad = max_frames - total_frames
            frame_indices = np.concatenate([
                frame_indices,
                np.full(pad, total_frames - 1),
            ])

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()
        return frames

    def generate(self, req: GenerationRequest) -> GeneratedClip:
        """Generate video using Luma Dream Machine API.

        Supports text-to-video natively.  Image-to-video requires a
        publicly accessible image URL (pass via ``extras['image_url']``)
        because the Dream Machine API rejects base64 payloads.
        """
        if not self.api_key:
            raise RuntimeError(
                "Luma adapter needs LUMA_API_KEY; configure in .env "
                "or use 'cinematic' backend for offline synthesis."
            )

        # Luma controls camera motion via prompt language.
        motion = req.extras.get("motion", req.extras.get("camera_motion", "camera_orbit"))
        style = req.extras.get("style", "cinematic")
        prompt = self._build_prompt(req.prompt, style, motion)

        # Resolve image URL for image-to-video.
        image_url = req.extras.get("image_url")
        if not image_url and req.first_frame is not None:
            # first_frame is typically a numpy array; Luma needs a public URL.
            # Save locally for reference, but the user must host it.
            temp_dir = os.path.join("outputs", "luma", "temp")
            saved = self._save_first_frame(req.first_frame, temp_dir)
            if saved:
                raise RuntimeError(
                    "Luma I2V requires a publicly accessible image URL. "
                    f"Saved first_frame to {saved}; upload it to a CDN and "
                    "pass image_url=<your_url> in your request."
                )

        # Resolve resolution, aspect ratio, and duration.
        resolution = self._resolution_for(req.width, req.height)
        aspect_ratio = self._aspect_ratio_for(req.width, req.height)
        duration_sec = min(max(int(req.duration), 2), 10)

        luma_req = LumaRequest(
            prompt=prompt,
            model=req.extras.get("model", "ray-2"),
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration_sec,
            loop=req.extras.get("loop", False),
            image=image_url,
            negative_prompt=req.negative_prompt,
            motion=motion,
            seed=req.seed,
            style_preset=req.extras.get("style_preset"),
        )

        task_id = self._submit_task(luma_req)
        print(f"Luma task submitted: {task_id}")
        task_data = self._poll_task(task_id)

        # Poll guarantees completion; task_data has state == "completed" or raised.

        # Extract video URL from assets (Luma returns it here).
        assets = task_data.get("assets", {})
        video_url = assets.get("video") or task_data.get("video_url")
        if not video_url:
            raise RuntimeError(
                f"No video URL in Luma response (state={task_data.get('state')}): {task_data}"
            )

        # Download the video
        temp_dir = os.path.join("outputs", "luma", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, f"luma_{task_id[:8]}.mp4")

        print("Downloading video from Luma...")
        self._download_video(video_url, output_path)
        print(f"Video downloaded to: {output_path}")

        # Extract frames
        frame_count = req.n_frames
        frames = self._download_frames(output_path, frame_count)

        if not frames:
            raise RuntimeError("No frames extracted from generated video")

        clip = GeneratedClip(
            prompt=req.prompt,
            backend="luma",
            frames=np.stack(frames),
            fps=req.fps,
            metadata={
                "task_id": task_id,
                "video_url": video_url,
                "motion": motion,
                "style": style,
                "model": luma_req.model,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "api_response": task_data,
            },
        )

        return clip
