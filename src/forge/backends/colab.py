"""Colab-hosted Wan backend - powerful realistic text-to-video AND image-to-video.

This laptop has no CUDA GPU, so realistic diffusion video is hosted on
``colab/CineForge_Colab_Video_Server.ipynb`` running a **Wan** model on a
Colab GPU. The notebook auto-selects a powerful 14B tier on an **A100**
(Colab Pro) and falls back to a 1.3B tier on a free T4, and it exposes two
endpoints this backend calls:

* ``POST /generate`` - text-to-video  (returns MP4 bytes)
* ``POST /image``    - image-to-video (multipart image + form fields)

Configure ``COLAB_BASE_URL`` in ``.env`` with the printed PUBLIC URL.
"""

from __future__ import annotations

import os
import uuid

import cv2
import numpy as np
import requests

from ..types import GeneratedClip, GenerationRequest
from .base import GeneratorBackend

# Prepended so users asking for "zombies on a street" get live-action
# footage by default instead of the animated look Wan also knows.
REALISM_SUFFIX = (
    "photorealistic live-action film footage, cinematic lighting, "
    "shot on digital cinema camera, natural motion"
)


def _safe_cast(value, cast, default):
    """Tolerant cast for optional headers."""
    try:
        if value is None:
            return default
        return cast(value)
    except (TypeError, ValueError):
        return default


class ColabBackend(GeneratorBackend):
    """Wan T2V + I2V served from a Colab GPU runtime via public URL."""

    name = "colab"

    def __init__(self, **env):
        super().__init__(**env)
        self.base_url = (os.getenv("COLAB_BASE_URL", "") or "").rstrip("/")
        self.timeout = int(os.getenv("COLAB_TIMEOUT", "3600"))

    def check(self) -> bool:
        if not self.base_url:
            return False
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            return r.status_code == 200
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------
    def _prompt(self, req) -> str:
        style = req.extras.get("style", "realistic")
        prompt = req.prompt or ""
        if style not in ("none", "raw") and prompt:
            prompt = (
                f"{prompt}, {REALISM_SUFFIX}"
                if style == "realistic"
                else f"{prompt}, {style} style"
            )
        return prompt.strip()

    def _params(self, req) -> dict:
        params = {
            "width": int(req.width),
            "height": int(req.height),
            "fps": min(int(req.fps), 30),
            "duration": float(min(req.duration, 10.0)),
            "num_inference_steps": int(req.extras.get("steps", 25)),
            "guidance_scale": float(req.extras.get("guidance_scale", 5.0)),
        }
        if req.seed is not None:
            params["seed"] = int(req.seed)
        return params

    def _decode_response(self, r, prompt: str, kind: str) -> GeneratedClip:
        if r.status_code != 200 or b"ftyp" not in r.content[:64]:
            raise RuntimeError(
                f"Colab {kind} generation failed: {r.status_code} {r.content[:300]!r}"
            )
        out_dir = os.path.join("outputs", "colab")
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, f"colab_{uuid.uuid4().hex[:8]}.mp4")
        with open(video_path, "wb") as f:
            f.write(r.content)
        frames = _decode_frames(video_path)
        if not frames:
            raise RuntimeError(f"no frames decoded from {video_path}")
        fps = _safe_cast(r.headers.get("X-Fps"), float, 24.0)
        return GeneratedClip(
            prompt=prompt,
            backend="colab",
            frames=np.stack(frames),
            fps=fps,
            metadata={
                "model": r.headers.get("X-Model", "wan"),
                "kind": kind,
                "enhanced_prompt": prompt,
                "seed": r.headers.get("X-Seed"),
                "elapsed_s": r.headers.get("X-Elapsed-S"),
                "video_path": video_path,
                "base_url": self.base_url,
            },
        )

    # ------------------------------------------------------------------
    def generate(self, req: GenerationRequest) -> GeneratedClip:
        """Text-to-video via the remote Wan text-to-video pipeline."""
        if not self.base_url:
            raise RuntimeError(
                "colab backend needs COLAB_BASE_URL: open "
                "colab/CineForge_Colab_Video_Server.ipynb on a Colab GPU "
                "(A100 recommended) and copy the printed URL into .env"
            )
        payload = {"prompt": self._prompt(req), "negative_prompt": req.negative_prompt or ""}
        payload.update(self._params(req))
        try:
            r = requests.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"could not reach Colab server at {self.base_url} - is it still "
                f"running? ({exc})"
            ) from exc
        clip = self._decode_response(r, req.prompt, "t2v")
        clip.metadata["requested_fps"] = int(req.fps)
        return clip

    def image_to_video(self, image_path, req: GenerationRequest) -> GeneratedClip:
        """Image-to-video: upload a still image and let the Wan I2V pipeline move it."""
        if not self.base_url:
            raise RuntimeError(
                "colab backend needs COLAB_BASE_URL: open the CineForge Colab "
                "notebook and copy its printed URL into .env"
            )
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)
        prompt = self._prompt(req)
        data = {"prompt": prompt, "negative_prompt": req.negative_prompt or ""}
        data.update(self._params(req))
        with open(image_path, "rb") as fh:
            files = {"file": (os.path.basename(image_path), fh, "image/png")}
            try:
                r = requests.post(
                    f"{self.base_url}/image", files=files, data=data, timeout=self.timeout
                )
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"could not reach Colab server at {self.base_url} - is it "
                    f"still running? ({exc})"
                ) from exc
        clip = self._decode_response(r, prompt, "i2v")
        clip.metadata["source_image"] = os.path.basename(image_path)
        return clip


def _decode_frames(path: str) -> list:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames
