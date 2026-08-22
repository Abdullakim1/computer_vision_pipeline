"""Prompt engineering: expands an idea into detailed, cinematic shots.

Deterministic (seeded) so demos are reproducible. Turns e.g. "aurora over a
valley" into shot specs with framing, camera, grade, and a strong descriptor
negative prompt - the raw material a diffusion or procedural backend reads.
"""

from __future__ import annotations

import hashlib
import random

_ADJ = (
    "cinematic, volumetric light, shallow depth of field, ultra-detailed, "
    "epic, atmospheric, masterful composition, film grain, anamorphic"
)
_NEG = (
    "blurry, low quality, worst quality, jpeg artifacts, watermark, text, "
    "disfigured, distorted, cartoon, too bright, overexposed"
)
_FRAMINGS = ("wide establishing shot", "slow tracking shot", "medium shot",
             "aerial vista", "proximity pull", "high-angle establishing")
_CAMERAS = ("pan", "orbit", "zoom", "dolly")
_LOOKS = ("teal-orange", "golden", "cyber", "noir", "vintage")
_LIGHTS = ("soft golden light", "hard blue moonlight", "caustic neon glow",
           "dusty backlight", "frosted morning light")


def _seed_int(seed) -> int:
    if isinstance(seed, int):
        return seed
    return int(hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:8], 16)


class CinematicPrompter:
    """Seeded shot-builder for the studio's 'direct a story' entry point."""

    def __init__(self, seed=None):
        self._seed = _seed_int(seed)

    def expand(self, idea: str, seed=None) -> dict:
        rng = random.Random(self._seed if seed is None else _seed_int(seed))
        framing = rng.choice(_FRAMINGS)
        camera = rng.choice(_CAMERAS)
        look = rng.choice(_LOOKS)
        light = rng.choice(_LIGHTS)
        prompt = f"{idea}, {framing}, {light}, {_ADJ}"
        return {
            "prompt": prompt,
            "negative_prompt": _NEG,
            "shot": {"framing": framing, "camera": camera, "light": light},
            "look": look,
            "seed": seed if seed is not None else self._seed,
        }

    def script(self, story: str, beats=3, seed=None) -> list:
        """Break a one-sentence story into ``beats`` distinct shot specs."""
        rng = random.Random(self._seed if seed is None else _seed_int(seed))
        words = story.split()
        shots = []
        for _ in range(beats):
            k = max(2, len(words) // beats)
            idea = " ".join(rng.sample(words, min(k, len(words))))
            shots.append(self.expand(idea, seed=rng.randrange(1 << 16)))
        return shots