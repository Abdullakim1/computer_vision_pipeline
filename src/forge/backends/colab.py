"""Colab-hosted Wan 2.1 backend — free realistic text-to-video.

This laptop has no CUDA GPU, so realistic diffusion video is hosted on a
free Google Colab T4 runtime by ``colab/CineForge_Colab_Video_Server.ipynb``
(Wan-AI/Wan2.1-T2V-1.3B, open weights). That notebook serves a small HTTP
API over a public proxy URL which needs no signup or token.

Configure with ``COLAB_BASE_URL`` in ``.env`` (the URL the notebook prints).
Prompts are automatically steered toward photorealistic live-action look,
the opposite of the procedural ``cinematic`` backend's animated style.
"""

from __future__ import annotations

import os
import tempfile
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


class ColabBackend(GeneratorBackend):
    """Wan 2.1 T2V served from a free Colab GPU runtime via public URL."""

    name = "colab"

    def __init__(self, **env):
        super().__init__(**env)
        self.base_url = (os.getenv("COLAB_BASE_URL", "") or "").rstrip("/")
        self.timeout = int(os.getenv("COLAB_TIMEOUT", "1800"))

    def check(self) -> bool:
        if not self.base_url:
            return False
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, req: GenerationRequest) -> GeneratedClip:
        if not self.base_url:
            raise RuntimeError(
                "colab backend needs COLAB_BASE_URL — open "
                "colab/CineForge_Colab_Video_Server.ipynb on a free Colab "
                "T4 runtime and copy the printed URL into .env"
            )

        style = req.extras.get("style", "realistic")
        prompt = req.prompt
        if style not in ("none", "raw"):
            prompt = f"{prompt}, {REALISM_SUFFIX}" if style == "realistic" else f"{prompt}, {style} style"

        payload = {
            "prompt": prompt,
            "negative_prompt": req.negative_prompt,
            # Wan's native 16:9 grid is 832x480; other ratios get snapped
            # to a multiple of 16 by the server.
            "width": int(req.width),
            "height": int(req.height),
            "fps": min(int(req.fps), 24),
            "duration": float(min(req.duration, 10.0)),
            "num_inference_steps": int(req.extras.get("steps", 25)),
            "guidance_scale": float(req.extras.get("guidance_scale", 5.0)),
        }
        if req.seed is not None:
            payload["seed"] = int(req.seed)

        try:
            r = requests.post(
                f"{self.base_url}/generate", json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"could not reach Colab server at {self.base_url} — is the "
                f"notebook still running? ({exc})"
            ) from exc
        if r.status_code != 200 or b"ftyp" not in r.content[:64]:
            raise RuntimeError(
                f"Colab generation failed: {r.status_code} {r.content[:300]!r}"
            )

        out_dir = os.path.join("outputs", "colab")
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, f"colab_{uuid.uuid4().hex[:8]}.mp4")
        with open(video_path, "wb") as f:
            f.write(r.content)

        frames = _decode_frames(video_path)
        if not frames:
            raise RuntimeError(f"no frames decoded from {video_path}")

        return GeneratedClip(
            prompt=req.prompt,
            backend="colab",
            frames=np.stack(frames),
            fps=req.fps,
            metadata={
                "model": "Wan-AI/Wan2.1-T2V-1.3B",
                "enhanced_prompt": prompt,
                "seed": r.headers.get("X-Seed"),
                "elapsed_s": r.headers.get("X-Elapsed-S"),
                "video_path": video_path,
                "base_url": self.base_url,
            },
        )


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
