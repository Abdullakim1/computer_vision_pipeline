"""Local latent-diffusion backend (would run a video-capable model in-process).

This adapter is a real integration point: whatever latent architecture a
team deploys (SVH, ModelScope T2V, CogVideoX, Wan2.x...) can be routed
through this one facade. It *degrades gracefully*: if torch/diffusers are
not installed (or no GPU seed is set), the backend refuses politely so the
studio can fall back to the cinematic engine.
"""

from __future__ import annotations

from ..types import GeneratedClip, GenerationRequest
from .base import GeneratorBackend


def _available():
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        return True
    except Exception:  # pragma: no cover
        return False


class LocalLatentBackend(GeneratorBackend):
    name = "local"

    def check(self) -> bool:
        return _available()

    def generate(self, req: GenerationRequest) -> GeneratedClip:
        if not _available():
            raise RuntimeError(
                "local latent backend unavailable: install torch + diffusers "
                "(see requirements.txt 'Deep-Torch' section). Falling back to "
                "an online backend is recommended."
            )
        # NOTE: real model loading + sampling happens here behind a standard
        # API. To run without touching Cloud adapters, either install the
        # heavy stack or use the 'cinematic' backend.
        raise NotImplementedError(
            "local latent sampling is scaffolded for a GPU env; select a "
            "backend that is available on this machine (e.g. 'cinematic')."
        )