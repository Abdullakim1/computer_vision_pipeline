"""Backend abstraction for text/video-to-video generation.

A ``GeneratorBackend`` is a strategy that turns a :class:`GenerationRequest`
into a :class:`GeneratedClip`. CineForge ships several backends mirroring how
real studios switch between clouds and local sampling:

* ``cinematic``  - always-online procedural cinematography (no deps).
* ``local``      - local latent diffusion (torch + diffusers) when installed.
* ``seedance``   - ByteDance Seedance (Volcano Ark) HTTP adapter.
* ``kling``      - Klong AI HTTP adapter.
"""

from __future__ import annotations

import abc
import os

from ..types import GeneratedClip, GenerationRequest


class GeneratorBackend(abc.ABC):
    """Uniform interface every synthesizer must satisfy."""

    name: str = "base"

    def __init__(self, **env):
        self.env = env or {}

    def __repr__(self):
        return f"<{self.__class__.__name__} [{self.name}]>"

    @property
    def kind(self) -> str:
        return self.name

    def check(self) -> bool:
        """Return True if this backend considers itself ready to run."""
        return True

    @abc.abstractclassmethod
    def generate(self, req: GenerationRequest) -> GeneratedClip:  # pragma: no cover
        raise NotImplementedError

    # ---- small shared helpers
    @staticmethod
    def _env(name: str, default=None):
        return os.getenv(name, default)