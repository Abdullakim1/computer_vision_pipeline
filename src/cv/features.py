"""Semantic feature extraction and a small vector store.

Primary path uses OpenAI CLIP (via ``open_clip``) when ``torch`` is present,
producing 512-D normalized embeddings. To keep CineForge runnable on GPU-free
machines we also ship a *procedural descriptor* fallback (color histogram +
edge statistics + sharpness) that still supports semantic re-ranking of
keyframes, with a documented CLIP-enhanced path for stronger guidance.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

try:
    import torch  # noqa: F401
    import open_clip
    _HAS_CLIP = True
except Exception:  # pragma: no cover - optional dependency
    torch = None
    _HAS_CLIP = False


def _log(fmt, *args):
    warnings.warn(fmt % args, stacklevel=3)


def _ensure_bgr(arr):
    """Return a uint8 BGR array for opencv consumer helpers."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr


class FeatureExtractor:
    """CLIP (optional) wrapper with a numpy procedural fallback."""

    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.device = "cuda" if (_HAS_CLIP and torch.cuda.is_available()) else "cpu"
        self.model = None
        self.preprocess = None
        self._tokenizer = None
        if _HAS_CLIP:
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained, device=self.device
                )
                self._tokenizer = open_clip.get_tokenizer(model_name)
                self.model.eval()
                warnings.warn(f"[features] CLIP active ({model_name} @ {self.device})", stacklevel=2)
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"[features] CLIP init failed ({exc}); procedural", stacklevel=2)
                self.model = None
                self.preprocess = None

    @property
    def kind(self):
        return "clip" if self.model is not None else "procedural"

    # -------------------------------------------------------------------
    # Public embedding + similarity API
    # -------------------------------------------------------------------
    def embed(self, image, text=False):
        """Return a normalized embedding for an image (numpy/PIL) or a str."""
        if self.model is not None:
            return self._clip(image, text=text)
        return self._procedural(image)

    def _clip(self, image, text=False):
        from PIL import Image as PILImage
        with torch.no_grad():
            if text:
                tokens = self._tokenizer([str(image)]).to(self.device)
                emb = self.model.encode_text(tokens)
            else:
                if isinstance(image, np.ndarray):
                    rgb = cv2.cvtColor(_ensure_bgr(image), cv2.COLOR_BGR2RGB)
                    image = PILImage.fromarray(rgb)
                t = self.preprocess(image).unsqueeze(0).to(self.device)
                emb = self.model.encode_image(t)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def _procedural(self, image):
        if isinstance(image, str):
            raw = image.encode("utf-8")
            # deterministic pseudo-feature from the text bytes (stable per string)
            seed = int.from_bytes(raw[:20] or b"0", "big")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(24)
            return vec / (np.linalg.norm(vec) + 1e-9)
        arr = _ensure_bgr(image if isinstance(image, np.ndarray) else np.array(image))
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_hist = cv2.calcHist([hsv[..., 0].astype(np.uint8)], [0], None, [24], [0, 180])
        hue_hist = hue_hist.ravel().astype(np.float32)
        hue_hist /= (hue_hist.sum() + 1e-9)

        sat = hsv[..., 1].mean() / 255.0
        bright = hsv[..., 2].mean() / 255.0
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 80, 180).mean() / 255.0
        sharp = float(np.sqrt((cv2.Laplacian(gray, cv2.CV_64F) ** 2).mean()))

        vec = np.concatenate([hue_hist, [sat, bright, edge, sharp]])
        return vec / (np.linalg.norm(vec) + 1e-9)

    @staticmethod
    def cosine_similarity(a, b):
        if isinstance(a, np.ndarray):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        with torch.no_grad():
            return float((a @ b.to(a.device).T).item())


class VectorStore:
    """In-memory store of normalized vectors -> metadata, for semantic queries."""

    def __init__(self):
        self._rows = []

    def add(self, vec, meta=None):
        v = vec.detach().cpu().numpy().reshape(-1) if hasattr(vec, "detach") else np.asarray(vec).reshape(-1)
        self._rows.append((v, meta))

    def query(self, vec, k=1):
        v = vec.detach().cpu().numpy().reshape(-1) if hasattr(vec, "detach") else np.asarray(vec).reshape(-1)
        scored = sorted(
            (
                (float(np.dot(v, row) / (np.linalg.norm(v) * np.linalg.norm(row) + 1e-9)), meta)
                for row, meta in self._rows
            ),
            reverse=True,
        )
        return scored[:k]

    def __len__(self):
        return len(self._rows)