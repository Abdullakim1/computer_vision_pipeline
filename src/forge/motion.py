"""Optical-flow frame interpolation / motion re-timing.

Imports dense flow via OpenCV's Lanczos-Warper (Far3ebeneck) approach and
warps/interleaves to synthetically raise the frame rate of any clip and to
generate in-between "motional" frames during image-to-video. GPU-free.
"""

from __future__ import annotations

import cv2
import numpy as np


def _flow(from_gray, to_gray):
    return cv2.calcOpticalFlowFarneback(
        from_gray, to_gray, None,
        pyr_scale=0.5, levels=5, winsize=21, iterations=5,
        poly_n=7, poly_sigma=1.5, flags=0,
    )


def _warp_forward(img, flow, alpha):
    """Advect ``img`` along ``flow`` scaled by ``alpha``."""
    h, w = img.shape[:2]
    fx = flow[..., 0].astype(np.float32)
    fy = flow[..., 1].astype(np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x.astype(np.float32) + alpha * fx).astype(np.float32)
    map_y = (grid_y.astype(np.float32) + alpha * fy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def interpolate_pair(frame_a, frame_b, n_steps=1, weights=None):
    """Produce ``n_steps`` in-between frames between A and B.

    Parameters
    ----------
    frame_a, frame_b : (H,W,3) uint8 BGR arrays.
    n_steps : number of interpolated frames to return.
    weights : optional per-step flow fractions in [0,1]; defaults to
        uniform 1/(n_steps+1)..n_steps/(n_steps+1).

    Returns
    -------
    list of (H,W,3) uint8 BGR in-between frames.
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    fwd = _flow(gray_a, gray_b)     # where A pixels move toward B
    bwd = _flow(gray_b, gray_a)     # where B pixels came from

    if weights is None:
        weights = [(i + 1) / (n_steps + 1) for i in range(n_steps)]
    out = []
    for t in weights:
        warp_a = _warp_forward(frame_a, fwd, t)
        warp_b = _warp_forward(frame_b, bwd, 1 - t)
        blended = ((1 - t) * warp_a + t * warp_b).astype(np.uint8)
        out.append(blended)
    return out


def interpolate_clip(frames, factor=2):
    """Lift a sequence by ``factor``x (frames as uint8 BGR, HxWx3)."""
    out = []
    for i in range(len(frames) - 1):
        out.append(frames[i])
        step = factor - 1
        if step > 0:
            out.extend(interpolate_pair(frames[i], frames[i + 1], step))
    out.append(frames[-1])
    return out


def ken_burns(image_rgb, w=720, h=1280, frames=None, dur=2.0, fps=24):
    """Slow push-in (Ken Burns) over a single input image, uint8 RGB in/out.

    Returns a list of uint8 BGR frames for easy reuse in the studio.
    """
    img = image_rgb if image_rgb.dtype == np.uint8 else np.clip(image_rgb * 255, 0, 255).astype(np.uint8)
    H, W = img.shape[:2]
    n = frames or int(dur * fps)
    cw, ch = min(W, H), min(W, H)
    out = []
    for i in range(n):
        t = i / max(1, n - 1)
        crop = int(ch * (0.5 + 0.35 * t))  # shrinking crop = push-in illusion
        cx = W // 2
        cy = H // 2
        half = max(1, crop // 2)
        x1, y1 = max(0, cx - half), max(0, cy - half)
        pan_y = int(h * 0.3 * t)
        cropped = img[max(0, y1 - pan_y):y1 - pan_y + crop, x1:x1 + crop]
        out.append(cv2.resize(cropped, (int(w), int(h)), interpolation=cv2.INTER_AREA))
    # OpenCV writes BGR; caller handles channel order on the way out.
    return out