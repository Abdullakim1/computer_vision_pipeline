"""Semantic conditioning - steering generation with vision intelligence.

Reuses the CLIP-style feature space (``src.cv.features``) to make generation
perception-aware: steer a sampler toward a target text/image, re-rank
candidate keyframes, and build a storyboard whose palette/pace drives the
look. Works in CLIP mode (torch present) or the procedural descriptor mode.
"""

from __future__ import annotations

import numpy as np

from src.cv.features import FeatureExtractor


class SemanticGuide:
    """Holds a guiding embedding (text or image) and scores candidates."""

    def __init__(self):
        self.extractor = FeatureExtractor()
        self.guide_vec = None
        self.guide_kind = None

    def set_text(self, text: str):
        self.guide_vec = self.extractor.embed(text, text=True)
        self.guide_kind = "text"
        return self

    def set_image(self, image):
        self.guide_vec = self.extractor.embed(image)
        self.guide_kind = "image"
        return self

    def score(self, image) -> float:
        if self.guide_vec is None:
            return 0.5
        vec = self.extractor.embed(image)
        return FeatureExtractor.cosine_similarity(self.guide_vec, vec)

    def rank_frames(self, frames) -> list:
        """Return (index, score) sorted by descending similarity to guide."""
        scored = [(i, self.score(f)) for i, f in enumerate(frames)]
        return sorted(scored, key=lambda x: x[1], reverse=True)


def interpolate_features(v, w, alpha):
    """Vector-space negotiation between two conditions (fmfm)."""
    return (1 - alpha) * v + alpha * w


def semantic_delta(short_prompt: str, rich_prompt: str) -> np.ndarray:
    """Direction vector pointing from 'short' toward 'rich' in embedding space."""
    fx = FeatureExtractor()
    a = fx.embed(short_prompt, text=True)
    b = fx.embed(rich_prompt, text=True)
    return _to_array(b) - _to_array(a)


def _to_array(vec):
    return vec if isinstance(vec, np.ndarray) else vec.detach().cpu().numpy().reshape(-1)