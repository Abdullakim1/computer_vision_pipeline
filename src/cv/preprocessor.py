"""Letterbox resize + normalize with correct box coordinate scaling."""

from __future__ import annotations

import cv2
import numpy as np


class Preprocessor:
    """Letterboxes a frame to a target size and returns scale/pad metadata.

    The returned ``mapping`` lets downstream consumers transform model-space
    coordinates back to the original frame, which is necessary when feature
    extraction happens on full-resolution crops.
    """

    def __init__(self, output_size=(640, 640)):
        w, h = output_size
        assert w and h, "output_size must be (width, height) > 0"
        self.output_size = (int(w), int(h))

    def process(self, frame):
        """Return ``(rgb01, mapping)``.

        ``mapping`` is a dict with ``scale``(factor), ``pad_w``, ``pad_h`` and
        ``original_shape`` so downstream can invert the transform.
        """
        in_h, in_w = frame.shape[:2]
        target_w, target_h = self.output_size

        scale = min(target_w / in_w, target_h / in_h)
        new_w = max(1, int(round(in_w * scale)))
        new_h = max(1, int(round(in_h * scale)))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        norm = rgb.astype(np.float32) / 255.0
        return norm, {
            "scale": scale,
            "pad_w": pad_w,
            "pad_h": pad_h,
            "original_shape": (in_h, in_w),
        }

    def unpad_coords(self, coords, mapping):
        """Undo letterbox padding on Nx2 ``(x1, y1 ...)`` arrays."""
        scale = mapping["scale"]
        pad_w, pad_h = mapping["pad_w"], mapping["pad_h"]
        return (np.asarray(coords) - np.array([pad_w, pad_h])) / scale