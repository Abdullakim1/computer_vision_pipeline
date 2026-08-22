"""CineForge perception stack.

Reusable computer-vision modules that provide the *vision intelligence*
(the conditioning layer) for the generative studio: objects are detected,
tracked across time, and embedded into a semantic vector space via CLIP so
generation can be steered by real scene understanding.

This package is backend-agnostic and dependency-graded: the detector
requires ``onnxruntime`` (optional), while the procedural feature path falls
back to lightweight descriptors automatically.
"""

from .streamer import Streamer
from .preprocessor import Preprocessor
from .features import FeatureExtractor, VectorStore

__all__ = [
    "Streamer",
    "Preprocessor",
    "FeatureExtractor",
    "VectorStore",
]