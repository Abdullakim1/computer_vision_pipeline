"""OpenCV drawing helpers for the vision layer (boxes, tracks, HUD, heatmaps)."""

from __future__ import annotations

import cv2
import numpy as np

# stable palette
_COLORS = np.random.RandomState(2024).randint(0, 255, (80, 3)).tolist()


def draw_detections(frame_bgr, detections, thickness=2):
    for det in detections:
        box = det.get("box") or ()
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = map(int, box)
        cid = det.get("class_id")
        color = tuple(_COLORS[cid % len(_COLORS)]) if cid is not None else (0, 255, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)

        parts = []
        if det.get("track_id") is not None:
            parts.append(f"ID {int(det['track_id'])}")
        parts.append(det.get("label") or (f"cls{cid}" if cid is not None else "obj"))
        if det.get("score") is not None:
            parts.append(f"{det['score']:.2f}")
        if det.get("similarity") is not None:
            parts.append(f"sim {det['similarity']:.2f}")
        label = " | ".join(parts)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(y1 - th - base, 0)
        cv2.rectangle(frame_bgr, (x1, y0), (x1 + tw, y0 + th), color, -1)
        cv2.putText(frame_bgr, label, (x1, y0 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame_bgr


def draw_hud(frame_bgr, lines, where="top-left"):
    """Overlay sensor HUD lines (frames processed, fps, mode ...)."""
    y = 20
    for txt in lines:
        cv2.putText(frame_bgr, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 22
    return frame_bgr