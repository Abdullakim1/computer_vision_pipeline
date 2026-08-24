"""Google Veo backend - Gemini API video generation.

Text-to-video via Google's Veo 3.1 models through the Gemini API
(generativelanguage.googleapis.com).  Generation happens entirely on
Google's GPUs — no local hardware needed.

Reads GEMINI_API_KEY from env; otherwise the adapter self-reports
unavailable.

API reference: https://ai.google.dev/gemini-api/docs/video
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests

from ..types import GeneratedClip, GenerationRequest
from .base import GeneratorBackend


class VeoBackend(GeneratorBackend):
    """Google Veo cloud backend (text-to-video, audio included)."""

    name = "veo"

    # Models available on the Gemini API (override via extras['model']
    # or the VEO_MODEL env var).
    MODELS = (
        "veo-3.1-fast-generate-preview",   # fast/cheap, free-tier friendly
        "veo-3.1-generate-preview",        # highest quality
        "veo-3.1-lite-generate-preview",   # light
    )

    ASPECT_RATIOS = {"16:9", "9:16"}

    def __init__(self, **env):
        super().__init__(**env)
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.base_url = os.getenv(
            "VEO_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        )
        self.model = os.getenv("VEO_MODEL", self.MODELS[0])
        self.timeout = int(os.getenv("VEO_TIMEOUT", "600"))
        self.poll_interval = int(os.getenv("VEO_POLL_INTERVAL", "10"))

    def check(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------
    def _submit(self, prompt: str, negative_prompt: str,
                aspect_ratio: str) -> str:
        """Submit a predictLongRunning request; return the operation name."""
        url = f"{self.base_url}/models/{self.model}:predictLongRunning"
        payload: Dict[str, Any] = {
            "instances": [{"prompt": prompt}],
            "parameters": {"aspectRatio": aspect_ratio},
        }
        if negative_prompt:
            payload["parameters"]["negativePrompt"] = negative_prompt

        resp = requests.post(
            url,
            json=payload,
            headers={"x-goog-api-key": self.api_key},
            timeout=60,
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Veo submit failed: {resp.status_code} - {resp.text[:500]}"
            )
        name = resp.json().get("name")
        if not name:
            raise RuntimeError(f"Veo response missing operation name: {resp.text[:500]}")
        return name

    # ------------------------------------------------------------------
    def _poll(self, operation: str) -> Dict[str, Any]:
        """Poll the long-running operation until done; return its JSON."""
        url = f"{self.base_url}/{operation}"
        start = time.time()
        while True:
            if time.time() - start > self.timeout:
                raise RuntimeError(f"Veo operation {operation} timed out")
            try:
                resp = requests.get(
                    url, headers={"x-goog-api-key": self.api_key}, timeout=30
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("done"):
                        if "error" in data:
                            err = data["error"].get("message", data["error"])
                            raise RuntimeError(f"Veo generation failed: {err}")
                        return data
                # else: transient error — fall through and retry
            except requests.exceptions.RequestException:
                pass
            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    def _download(self, file_uri: str, output_path: str) -> str:
        """Download the generated video file (URI includes the API token)."""
        url = f"{self.base_url}/{file_uri}:download?alt=media"
        with requests.get(
            url, headers={"x-goog-api-key": self.api_key},
            stream=True, timeout=300,
        ) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        return output_path

    @staticmethod
    def _extract_frames(video_path: str, max_frames: int) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"No frames found in video: {video_path}")
        if total >= max_frames:
            indices = np.linspace(0, total - 1, max_frames, dtype=int)
        else:
            indices = np.arange(total)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    # ------------------------------------------------------------------
    def generate(self, req: GenerationRequest) -> GeneratedClip:
        if not self.api_key:
            raise RuntimeError(
                "Veo adapter needs GEMINI_API_KEY; configure it in .env."
            )

        model = req.extras.get("model", self.model)
        self.model = model

        aspect = req.extras.get("aspect_ratio")
        if not aspect:
            aspect = "9:16" if req.height > req.width else "16:9"

        operation = self._submit(req.prompt, req.negative_prompt or "", aspect)
        print(f"Veo task submitted: {operation}")
        result = self._poll(operation)
        print("Veo generation complete; downloading video...")

        response = result.get("response", {})
        rai_reasons = response.get("raiMediaFilteredReasons")
        generated = response.get("generatedSamples") or response.get(
            "generatedVideos"
        ) or []
        if not generated:
            raise RuntimeError(
                "Veo returned no video samples"
                + (f" (filtered: {rai_reasons})" if rai_reasons else "")
            )
        video = generated[0].get("video", {})
        file_uri = video.get("uri") or video.get("gcsUri")
        if not file_uri:
            raise RuntimeError(f"Veo response missing video URI: {response}")

        temp_dir = os.path.join("outputs", "veo")
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(
            temp_dir, f"veo_{operation.split('/')[-1][:12]}.mp4"
        )
        self._download(file_uri, output_path)
        print(f"Video downloaded to: {output_path}")

        frames = self._extract_frames(output_path, req.n_frames)
        if not frames:
            raise RuntimeError("No frames extracted from Veo video")

        return GeneratedClip(
            prompt=req.prompt,
            backend="veo",
            frames=np.stack(frames),
            fps=req.fps,
            metadata={
                "model": model,
                "operation": operation,
                "video_path": output_path,
                "aspect_ratio": aspect,
                "api_response": response,
            },
        )
