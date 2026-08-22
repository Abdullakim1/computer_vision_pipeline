"""Cinematic post: color grade, bloom, grain, vignette, letterbox.

A small but real film-look pipeline with GPU-free numpy/opencv so every clip
leaves the factory looking deliberate rather than raw.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# FilmLook — a guiding 'lot' asset
# ---------------------------------------------------------------------------
@dataclass
class FilmLook:
    name: str
    contrast: float = 1.1
    shadows_shift: tuple = (0.0, 0.0, 0.0)     # RGB lift on dark end
    highlights_shift: tuple = (0.0, 0.0, 0.0)   # RGB tint on bright end
    temp: float = 0.0                            # warm >0, cool <0
    vignette: float = 0.0
    grain: float = 0.0
    bloom: float = 0.0
    bokeh: float = 0.0
    chromatic: float = 0.0
    scanlines: float = 0.0
    vintage: float = 0.0
    sepia: float = 0.0

    @classmethod
    def presets(cls):
        """Get all film look presets."""
        from .effects import presets
        return presets()

    @property
    def names(cls):
        """Get all preset names."""
        from .effects import presets
        return list(presets().keys())



def presets():
    return {
        "argo": FilmLook("argo", contrast=1.06, vignette=0.18, grain=0.0),
        "teal-orange": FilmLook(
            "teal-orange", contrast=1.16, vignette=0.35, grain=0.02,
            shadows_shift=(-0.015, 0.0, -0.04), highlights_shift=(0.03, 0.0, 0.0),
        ),
        "cyber": FilmLook(
            "cyber", contrast=1.28, vignette=0.42, grain=0.012, bloom=0.12,
            shadows_shift=(-0.05, 0.03, 0.07),
        ),
        "golden": FilmLook(
            "golden", contrast=1.02, vignette=0.28, grain=0.015, temp=0.18,
            shadows_shift=(0.05, 0.01, -0.06),
        ),
        "noir": FilmLook(
            "noir", contrast=1.32, vignette=0.5, grain=0.035,
            shadows_shift=(-0.04, 0.02, 0.03),
        ),
        "vintage": FilmLook(
            "vintage", contrast=0.9, vignette=0.3, grain=0.09,
            highlights_shift=(0.03, 0.02, -0.015),
        ),
        "widescreen": FilmLook("widescreen", contrast=1.12, vignette=0.25, grain=0.015),
        "cinematic": FilmLook("cinematic", contrast=1.1, vignette=0.3, grain=0.02),
        "dramatic": FilmLook("dramatic", contrast=1.25, vignette=0.45, grain=0.025),
        "soft": FilmLook("soft", contrast=0.95, vignette=0.15, grain=0.005),
        "high_contrast": FilmLook("high_contrast", contrast=1.35, vignette=0.35, grain=0.01),
        "cold": FilmLook("cold", contrast=1.08, vignette=0.2, temp=-0.12, grain=0.01),
        "warm": FilmLook("warm", contrast=1.05, vignette=0.2, temp=0.15, grain=0.01),
        "grayscale": FilmLook("grayscale", contrast=1.0, grain=0.015, scanlines=0.1),
        "tear": FilmLook("tear", contrast=1.2, vignette=0.4, chromatic=1.5),
        "dreamy": FilmLook("dreamy", contrast=1.0, vignette=0.25, bloom=0.2),
    }


def lookup(name: str) -> FilmLook:
    return presets().get(name, presets()["argo"])


# ---------------------------------------------------------------------------
# Low-level ops (float32 in [0,255])
# ---------------------------------------------------------------------------
def _lift_gain(img, shadows: tuple, highlights: tuple):
    out = img.astype(np.float32)
    for c in range(3):
        out[..., c] *= (1.0 + shadows[c] + highlights[c])
    out = np.clip(out, 0, 255)
    return out


def _contrast(img: np.ndarray, c: float) -> np.ndarray:
    x = img / 255.0 - 0.5
    x = x * c
    return np.clip((x + 0.5) * 255.0, 0, 255)


def _mild_curve(img: np.ndarray) -> np.ndarray:
    """Soft S-curve via AOV on the 0..1 axis."""
    x = np.clip(img.astype(np.float32) / 255.0, 0, 1)
    s = np.clip(1.5 * x * (2 - x), 0, 1)
    s = x * (1.0 + 0.35 * (x - s))          # gentle film toe/shoulder
    return np.clip(s * 255.0, 0, 255)


def _vignette(img: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w / 2) / (0.5 * w)) ** 2 + ((yy - h / 2) / (0.5 * h)) ** 2)
    mask = 1.0 - strength * np.clip(d, 0, 1.5) ** 2
    return np.clip(img * mask[..., None], 0, 255)


def _grain(img: np.ndarray, amount: float, seed) -> np.ndarray:
    if amount <= 0:
        return img
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, amount * 255.0, img.shape)
    return np.clip(img + noise, 0, 255)


def _bloom(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    lum = img.mean(axis=2)
    mask = np.clip((lum - 180.0) / 70.0, 0, 1)[..., None]
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=14)
    return np.clip(img + blur * mask * amount, 0, 255)


def _temp_shift(img: np.ndarray, temp: float) -> np.ndarray:
    if abs(temp) < 1e-6:
        return img
    out = img.astype(np.float32)
    out[..., 0] += temp * 18
    out[..., 2] -= temp * 18
    return np.clip(out, 0, 255)


def _letterbox(img: np.ndarray, ratio=0.05) -> np.ndarray:
    h, w = img.shape[:2]
    bar = int(h * ratio)
    out = np.zeros_like(img)
    out[bar:h - bar] = img[bar:h - bar]
    return out

# ---------------------------------------------------------------------------
# Advanced Effects
# ---------------------------------------------------------------------------
def _bokeh_effect(frame: np.ndarray, blur_radius: float = 15.0) -> np.ndarray:
    if blur_radius <= 0:
        return frame
    f = frame.astype(np.float32)
    luminance = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    blurred = cv2.GaussianBlur(f, (0, 0), blur_radius)
    dof_strength = 0.6
    out = (f * (1 - dof_strength) + blurred * dof_strength).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _chromatic_aberration(frame: np.ndarray, intensity: float = 2.0) -> np.ndarray:
    if intensity <= 0.1:
        return frame
    h, w = frame.shape[:2]
    f = frame.astype(np.float32)
    r = f[..., 0]
    g = f[..., 1]
    b = f[..., 2]
    shift = int(intensity)
    r_shift = np.roll(r, shift, axis=1)
    r_shift[:, :shift] = 0
    b_shift = np.roll(b, -shift, axis=1)
    b_shift[:, -shift:] = 0
    edge = np.abs(r - g) + np.abs(g - b)
    edge = (edge / edge.max() * intensity)[:, :, np.newaxis]
    r = np.clip(r_shift + edge[..., 0] * 0.5, 0, 255)
    b = np.clip(b_shift + edge[..., 2] * 0.5, 0, 255)
    out = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return out


def _scanlines(frame: np.ndarray, intensity: float = 0.15) -> np.ndarray:
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    scanline_pattern = np.sin(np.arange(h) * 0.5) * intensity
    scanline_pattern = scanline_pattern[:, np.newaxis, np.newaxis]
    out = frame.astype(np.float32)
    out = out * (1 - scanline_pattern)
    return np.clip(out, 0, 255).astype(np.uint8)


def _film_grain(frame: np.ndarray, amount: float = 0.03, seed: int = 42) -> np.ndarray:
    if amount <= 0:
        return frame
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, amount * 255, frame.shape)
    out = frame.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _letterbox_bar(frame: np.ndarray, ratio: float = 0.05) -> np.ndarray:
    h, w = frame.shape[:2]
    bar = int(h * ratio)
    out = np.zeros((h + bar * 2, w, 3), dtype=frame.dtype)
    out[bar:bar + h] = frame
    return out

# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def grade(frame: np.ndarray, look: FilmLook, seed=None, letterbox=False, **kwargs) -> np.ndarray:
    """Apply a full film look to a uint8 RGB frame. Returns uint8 RGB."""
    f = frame.astype(np.float32)
    f = _lift_gain(f, look.shadows_shift, look.highlights_shift)
    f = _contrast(f, look.contrast)
    f = _mild_curve(f)
    f = _temp_shift(f, look.temp)
    f = _grain(f, look.grain, seed or 42)
    f = _vignette(f, look.vignette)
    f = _bloom(f, look.bloom)
    f = _bokeh_effect(f, look.bokeh)
    f = _chromatic_aberration(f, look.chromatic)
    f = _scanlines(f, look.scanlines)
    out = np.clip(f, 0, 255).astype(np.uint8)
    return _letterbox_bar(out, ratio=0.05) if letterbox else out
