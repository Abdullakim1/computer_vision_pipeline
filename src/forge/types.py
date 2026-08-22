"""Common datatypes shared across generation backends and the studio."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class GenerationRequest:
    """Canonical request passed to every backend."""

    prompt: str = ""
    negative_prompt: str = "blurry, low quality, worst quality, jpeg artifacts"
    width: int = 1280
    height: int = 720
    fps: int = 24
    duration: float = 4.0
    seed: int | None = None
    first_frame: np.ndarray | None = None        # HxWx3 uint8 RGB, optional
    last_frame: np.ndarray | None = None
    reference_images: list = field(default_factory=list)
    motion_strength: float = 0.6
    extras: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return max(1, int(round(self.duration * self.fps)))


@dataclass
class GeneratedClip:
    """A finished piece of video: 4D uint8 RGB ``frames`` (T, H, W, 3)."""

    prompt: str
    backend: str
    frames: np.ndarray
    fps: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if getattr(self.frames, "ndim", 0) != 4 or (self.frames.shape[-1] != 3):
            raise ValueError("frames must be 4D (T, H, W, 3) uint8 RGB")
        self.T, self.height, self.width, _ = self.frames.shape

    @property
    def duration_s(self) -> float:
        """Playback duration of the clip in seconds."""
        return (self.T / self.fps) if self.fps else 0.0

    # ------------------------------------------------------------------
    def write_video(self, path, fps=None, codec=None) -> str:
        """Write RGB frames to a .mp4/.avi via OpenCV (BGR internally)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fps = fps or self.fps or 24
        if codec is None:
            codec = "mp4v" if p.suffix.lower() == ".mp4" else "XVID"
        bgr = np.ascontiguousarray(self.frames[:, :, :, ::-1])  # RGB -> BGR
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(p), fourcc, float(fps), (self.width, self.height))
        for f in bgr:
            writer.write(f)
        writer.release()
        return str(p)

    def to_gif(self, path, fps=12, scale=1.0) -> str:
        step = max(1, int(round(self.fps / fps)))
        frames = [Image.fromarray(f) for f in self.frames[::step]]
        if scale != 1.0:
            w = max(1, int(self.width * scale))
            h = max(1, int(self.height * scale))
            frames = [im.resize((w, h), Image.LANCZOS) for im in frames]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0)
        return path

    def contact_sheet(self, cols=4, rows=2) -> np.ndarray:
        """Composite frames into a single RGB contact sheet image."""
        t = self.T
        idx = np.linspace(0, t - 1, cols * rows).astype(int)
        canvas = np.zeros((self.height * rows, self.width * cols, 3), np.uint8)
        for r in range(rows):
            for c in range(cols):
                f = self.frames[idx[r * cols + c]]
                canvas[r * self.height:(r + 1) * self.height,
                       c * self.width:(c + 1) * self.width] = f
        return canvas