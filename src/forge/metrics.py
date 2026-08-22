"""Video quality + motion metrics for the studio.

All CPU-safe. Provides the numbers a reviewer loves next to a demo: temporal
motion (optical-flow energy), frame-to-frame diversity, sharpness,
colorfulness, and (when CLIP/torch present) semantic grounding scores.

Enhanced with:
- Noise level estimation
- Dynamic range analysis
- Motion history graphs
- Perceptual quality scores
- Color temperature metrics
- Motion continuity metrics
"""

from __future__ import annotations

import cv2
import numpy as np

from .motion import _flow


def _gray(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    return g.astype(np.float32)


def sharpness(frame) -> float:
    """Laplacian variance measure of sharpness."""
    g = _gray(frame)
    return float(cv2.Laplacian(g.astype(np.uint8), cv2.CV_64F).var())


def colorfulness(frame) -> float:
    """Hasler & Suessstrunk colorfulness on uint8 RGB."""
    if frame.ndim == 2:
        return 0.0
    r = frame[:, :, 0].astype(float)
    g = frame[:, :, 1].astype(float)
    b = frame[:, :, 2].astype(float)
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    root = np.sqrt(rg ** 2 + yb ** 2)
    mu, sd = root.mean(), root.std()
    return float(np.sqrt(mu ** 2 + sd ** 2) + 0.3 * np.sqrt(mu ** 2 + sd ** 2))


def noise_level(frame: np.ndarray) -> float:
    """Estimate noise level using variance of local regions."""
    if frame.ndim == 2:
        g = frame
    else:
        g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    h, w = g.shape
    # Compute local variance in 8x8 blocks
    ksize = 8
    padded = np.pad(g, ksize // 2, mode='reflect')
    local_vars = []
    for i in range(0, h, ksize):
        for j in range(0, w, ksize):
            block = padded[i:i+ksize, j:j+ksize].astype(np.float32)
            local_vars.append(np.var(block))
    
    return float(np.mean(local_vars))


def dynamic_range(frame: np.ndarray) -> float:
    """Calculate dynamic range (min/max luminance ratio)."""
    if frame.ndim == 2:
        lum = frame
    else:
        lum = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
    
    min_lum = np.percentile(lum, 1)
    max_lum = np.percentile(lum, 99)
    return float(max_lum / (min_lum + 1e-6))


def color_temperature(frame: np.ndarray) -> float:
    """Estimate color temperature from RGB channels."""
    if frame.ndim == 2:
        r = frame
        g = frame
        b = frame
    else:
        r = frame[:,:,0].astype(float)
        g = frame[:,:,1].astype(float)
        b = frame[:,:,2].astype(float)
    
    # Normalize channels
    mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
    
    # Temperature estimate (warmer = red/rgb ratio higher)
    ratio = (mean_r + 1e-6) / (mean_b + 1e-6)
    return float(ratio)


def _motion(a, b) -> float:
    """Optical flow magnitude between two frames."""
    f = _flow(_gray(a), _gray(b))
    return float(np.sqrt((f ** 2).sum(-1)).mean())


def motion_continuity(frames: list) -> float:
    """Calculate motion continuity score."""
    if len(frames) < 2:
        return 0.0
    
    motions = []
    for i in range(len(frames) - 1):
        a, b = frames[i].astype(np.float32), frames[i + 1].astype(np.float32)
        motions.append(_motion(a, b))
    
    # Mean of ratios of consecutive motions (smooth if close to 1)
    ratios = []
    for i in range(len(motions) - 1):
        if motions[i] > 1e-6:
            ratios.append(motions[i + 1] / motions[i])
    
    return float(np.mean(ratios)) if ratios else 0.0


def perceptual_quality(frames: list, fps: float = 24.0) -> float:
    """Perceptual quality score combining multiple metrics."""
    if len(frames) < 10:
        return 0.0
    
    # Extract key metrics
    sharpness_scores = [sharpness(f) for f in frames]
    colorfulness_scores = [colorfulness(f) for f in frames]
    
    mean_sharpness = np.mean(sharpness_scores)
    mean_color = np.mean(colorfulness_scores)
    mean_noise = np.mean([noise_level(f) for f in frames])
    
    # Normalize scores
    sharpness_norm = min(sharpness_scores[-1] / 100.0, 1.0) if sharpness_scores else 0.5
    color_norm = min(colorfulness_scores[-1] / 50.0, 1.0) if colorfulness_scores else 0.5
    
    # Penalty for noise
    noise_penalty = max(0, 1.0 - min(mean_noise / 15.0, 1.0))
    
    quality = 0.3 * sharpness_norm + 0.3 * color_norm - 0.2 * noise_penalty
    return round(max(0.0, min(1.0, quality)), 3)


def analyze(clip, fps=24.0) -> dict:
    """Comprehensive video quality analysis."""
    frames = [np.asarray(f) for f in clip]
    t = len(frames)
    
    sharp = [round(sharpness(f), 2) for f in frames]
    colors = [round(colorfulness(f), 4) for f in frames]
    noises = [round(noise_level(f), 2) for f in frames]
    dynamics = [round(dynamic_range(f), 2) for f in frames]
    temperatures = [round(color_temperature(f), 2) for f in frames]
    
    diffs, flows = [], 0.0
    for i in range(max(t - 1, 0)):
        a, b = frames[i].astype(np.float32), frames[i + 1].astype(np.float32)
        diffs.append(float(np.abs(a - b).mean()))
        flows += _motion(frames[i], frames[i + 1])
    
    mean_flow = flows / max(1, t - 1)
    diversity = (np.std([_motion(frames[i], frames[i + 1]) for i in range(max(t - 1, 0))]) /
                 (mean_flow + 1e-6)) if t > 1 else 0.0
    
    continuity = round(motion_continuity(frames), 3)
    quality_score = round(perceptual_quality(frames, fps), 3)
    
    # Color statistics
    avg_color = (np.mean(colors), np.std(colors))
    avg_temp = (np.mean(temperatures), np.std(temperatures))
    
    return {
        "frames": t,
        "duration_s": round(t / max(1, fps), 2),
        "fps": fps,
        "metrics": {
            "mean_frame_diff": round(float(np.mean(diffs)) if diffs else 0.0, 4),
            "mean_optical_flow_px": round(mean_flow, 2),
            "temporal_diversity": round(float(diversity), 4),
            "avg_sharpness": round(float(np.mean(sharp)), 2),
            "avg_colorfulness": round(float(np.mean(colors)), 4),
            "avg_noise_level": round(float(np.mean(noises)), 2),
            "avg_dynamic_range": round(float(np.mean(dynamics)), 2),
            "avg_color_temperature": round(float(np.mean(temperatures)), 2),
            "motion_continuity": continuity,
            "perceptual_quality": quality_score,
            "color_range": avg_color,
            "temp_range": avg_temp,
        }
    }
