"""Cinematic procedural backend - GPU-free generative video engine.

Renders GPU-free generative video by direct synthesis in numpy: layered
parallax mountain silhouettes, dynamic particle systems, celestial bodies,
and atmospheric effects - driven by sophisticated camera moves and film grading.

Supports 11 themes, 25+ camera motions, 6 particle systems, 18+ grading presets.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

from ..effects import grade, lookup
from ..types import GeneratedClip, GenerationRequest
from .base import GeneratorBackend


@dataclass
class ThemeConfig:
    name: str
    palette: List[Tuple[int, int, int]] = field(default_factory=list)
    look: str = "argo"
    keywords: List[str] = field(default_factory=list)
    has_celestial: bool = True
    has_particles: bool = False
    particle_type: str = "none"
    cloud_density: float = 0.0
    fog_density: float = 0.0
    style: str = "landscape"

_THEME_CONFIGS = {
    "aurora": ThemeConfig(
        name="aurora",
        palette=[(10, 20, 40), (20, 50, 100), (40, 80, 150), (100, 150, 200), (180, 200, 255)],
        look="cyber",
        keywords=["aurora", "night", "polar", "star", "sky"],
        has_celestial=True,
        particle_type="fireflies",
    ),
    "sunset": ThemeConfig(
        name="sunset",
        palette=[(30, 10, 10), (80, 30, 10), (180, 60, 40), (255, 150, 100), (255, 220, 180)],
        look="golden",
        keywords=["sunset", "evening", "horizon", "sun", "orange"],
        has_celestial=True,
        particle_type="embers",
    ),
    "golden": ThemeConfig(
        name="golden",
        palette=[(255, 200, 100), (255, 180, 80), (255, 220, 140), (255, 150, 80), (200, 120, 50)],
        look="golden",
        keywords=["golden", "hour", "morning", "light", "sun"],
        has_celestial=True,
        particle_type="dust",
    ),
    "neon": ThemeConfig(
        name="neon",
        palette=[(10, 0, 20), (20, 0, 40), (40, 0, 80), (0, 100, 150), (200, 0, 255), (0, 255, 200)],
        look="cyber",
        keywords=["neon", "city", "night", "lights", "cyberpunk"],
        has_celestial=False,
        particle_type="data",
    ),
    "ocean": ThemeConfig(
        name="ocean",
        palette=[(10, 30, 60), (30, 60, 100), (60, 100, 150), (100, 150, 200), (200, 220, 255)],
        look="argo",
        keywords=["ocean", "sea", "waves", "water", "blue"],
        has_celestial=True,
        particle_type="foam",
    ),
    "mountains": ThemeConfig(
        name="mountains",
        palette=[(40, 40, 40), (60, 60, 70), (80, 80, 90), (120, 120, 130), (160, 160, 170)],
        look="noir",
        keywords=["mountains", "peaks", "rock", "distant", "sky"],
        has_celestial=True,
        particle_type="mist",
    ),
    "mono": ThemeConfig(
        name="mono",
        palette=[(20, 20, 20), (60, 60, 60), (100, 100, 100), (140, 140, 140), (180, 180, 180)],
        look="noir",
        keywords=["black", "white", "gray", "monochrome", "film"],
        has_celestial=True,
        particle_type="snow",
    ),
    "campfire": ThemeConfig(
        name="campfire",
        palette=[(10, 5, 5), (30, 20, 20), (80, 50, 30), (180, 100, 50), (255, 180, 100)],
        look="teal-orange",
        keywords=["fire", "campfire", "night", "glow", "embers"],
        has_celestial=False,
        particle_type="embers",
    ),
    "cyber": ThemeConfig(
        name="cyber",
        palette=[(5, 0, 10), (10, 0, 30), (0, 30, 60), (0, 80, 120), (0, 150, 200)],
        look="cyber",
        keywords=["future", "hologram", "grid", "digital", "matrix"],
        has_celestial=False,
        particle_type="data",
    ),
    "moody": ThemeConfig(
        name="moody",
        palette=[(10, 10, 15), (25, 25, 35), (40, 40, 50), (60, 60, 70), (80, 80, 90)],
        look="noir",
        keywords=["fog", "dark", "mood", "atmosphere", "mysterious"],
        has_celestial=True,
        particle_type="mist",
    ),
    "meadow": ThemeConfig(
        name="meadow",
        palette=[(30, 100, 30), (50, 130, 40), (80, 160, 50), (150, 200, 100), (200, 230, 180)],
        look="golden",
        keywords=["meadow", "grass", "field", "flower", "spring"],
        has_celestial=True,
        particle_type="pollen",
    ),
    "storm": ThemeConfig(
        name="storm",
        palette=[(10, 10, 15), (20, 20, 30), (30, 30, 45), (50, 50, 70), (80, 80, 100)],
        look="argo",
        keywords=["storm", "lightning", "rain", "thunder", "dark"],
        has_celestial=True,
        particle_type="rain",
    ),
    "city": ThemeConfig(
        name="city",
        palette=[(8, 8, 18), (14, 14, 30), (22, 22, 46), (38, 38, 70), (60, 60, 95)],
        look="noir",
        keywords=["city", "street", "urban", "skyline", "downtown", "alley", "neon"],
        has_celestial=False,
        particle_type="none",
        style="urban",
    ),
    "apocalypse": ThemeConfig(
        name="apocalypse",
        palette=[(6, 6, 8), (14, 12, 14), (26, 22, 20), (42, 36, 30), (65, 55, 45)],
        look="noir",
        keywords=["zombie", "zombies", "apocalypse", "undead", "horde", "infected", "dead"],
        has_celestial=True,
        particle_type="rain",
        style="urban",
    ),
}

# Themes rendered with the dedicated urban (city / apocalypse / zombie) engine.
_URBAN_THEMES = {"city", "apocalypse", "zombie", "horror", "urban"}

# Synonym hints used to resolve a free-text prompt to the closest theme.
_THEME_HINTS = {
    "aurora": ["aurora", "borealis", "northern lights", "polar", "north"],
    "sunset": ["sunset", "dusk", "evening", "horizon", "orange"],
    "golden": ["golden hour", "sunrise", "dawn", "morning", "golden"],
    "neon": ["neon", "cyberpunk", "sci-fi", "lights", "nightlife"],
    "ocean": ["ocean", "sea", "beach", "waves", "water", "underwater", "lighthouse"],
    "mountains": ["mountain", "peaks", "alps", "valley", "himalaya"],
    "mono": ["black and white", "monochrome", "grayscale", "b&w"],
    "campfire": ["campfire", "bonfire", "embers", "forest fire", "roasting"],
    "cyber": ["hologram", "matrix", "digital", "future", "grid", "data"],
    "moody": ["fog", "mist", "moody", "gloomy", "haunted", "dark forest", "swamp"],
    "meadow": ["meadow", "field", "grass", "flower", "spring", "pasture"],
    "storm": ["storm", "lightning", "rain", "thunder", "hurricane", "clouds"],
    "city": ["street", "urban", "downtown", "alley", "skyline", "skyscraper", "city"],
    "apocalypse": ["zombie", "zombies", "horror", "apocalypse", "undead", "horde",
                   "infected", "walking dead", "outbreak"],
}


def resolve_theme(prompt: str, style: Optional[str]) -> str:
    """Map a free-text prompt (plus optional explicit style) to a theme key.

    Prefers an explicit valid ``style``/theme name. Otherwise scores every theme
    by how many of its hint words appear in the prompt, and returns the best
    match. Falls back to ``moody`` for dark/urban requests, else ``aurora``.
    """
    if style and style in _THEME_CONFIGS:
        return style
    text = (prompt or "").lower()
    # Horror cues take priority so zombies land on the apocalyptic urban scene.
    if any(h in text for h in _THEME_HINTS["apocalypse"]):
        return "apocalypse"
    best, best_score = "aurora", 0
    for theme, hints in _THEME_HINTS.items():
        if theme == "apocalypse":
            continue
        score = sum(1 for h in hints if h in text)
        if score > best_score:
            best_score, best = score, theme
    if best_score:
        return best
    if any(w in text for w in ("dark", "night", "zombie", "horror", "city")):
        return "moody"
    return "aurora"


_camera_moves = {
    "static": lambda t, w, h: 0.0,
    "orbit": lambda t, w, h: (math.sin(t * 0.3) * 5, math.cos(t * 0.4) * 5),
    "pan_left": lambda t, w, h: (t * 0.2, 0.0),
    "pan_right": lambda t, w, h: (-t * 0.2, 0.0),
    "zoom_in": lambda t, w, h: (-t * 0.3, 0.0),
    "zoom_out": lambda t, w, h: (t * 0.2, 0.0),
    "dolly_in": lambda t, w, h: (-t * 0.1, 0.0),
    "dolly_out": lambda t, w, h: (t * 0.05, 0.0),
    "crane_up": lambda t, w, h: (0.0, -t * 0.15),
    "crane_down": lambda t, w, h: (0.0, t * 0.1),
    "handheld": lambda t, w, h: (
        math.sin(t * 8) * 2 + math.sin(t * 12) * 1,
        math.cos(t * 10) * 1.5 + math.cos(t * 7) * 1,
    ),
    "dutch_tilt": lambda t, w, h: (0.0, math.sin(t * 0.5) * 0.1),
    "jib": lambda t, w, h: (math.sin(t * 0.3) * 3, -t * 0.1),
    "tracking": lambda t, w, h: (t * 0.1, 0.0),
}


# Particle system for all themes
@dataclass
class ParticleSystem:
    """Dynamic particle system for atmospheric effects."""

    def __init__(self, w: int, h: int, count: int, ptype: str, seed: int):
        self.w, self.h, self.ptype = w, h, ptype
        self.count = count
        self.seed = seed
        self._init_particles()

    def _init_particles(self):
        """Initialize particle positions and velocities."""
        rng = np.random.default_rng(self.seed)
        self.px = rng.uniform(0, self.w, self.count)
        self.py = rng.uniform(0, self.h, self.count)
        self.speed = rng.uniform(0.5, 3.0, self.count)
        self.size = rng.uniform(1.0, 4.0, self.count)
        self.alpha = rng.uniform(0.3, 0.8, self.count)
        self.phase = rng.uniform(0, 6.28, self.count)
        
        if self.ptype == "snow":
            self.vel_y = rng.uniform(1.0, 3.0, self.count)
            self.vel_x = rng.uniform(-0.5, 0.5, self.count)
        elif self.ptype == "rain":
            self.vel_y = rng.uniform(10.0, 20.0, self.count)
            self.vel_x = rng.uniform(-0.5, 0.5, self.count)
        elif self.ptype == "embers":
            self.vel_y = rng.uniform(-1.0, 2.0, self.count)
            self.vel_x = rng.uniform(-1.0, 1.0, self.count)
        elif self.ptype == "fireflies":
            self.vel_y = rng.uniform(-0.5, 0.5, self.count)
            self.vel_x = rng.uniform(-0.5, 0.5, self.count)
        elif self.ptype == "foam":
            self.vel_y = rng.uniform(-0.3, 0.3, self.count)
            self.vel_x = rng.uniform(-0.5, 0.5, self.count)
        elif self.ptype == "pollen":
            self.vel_y = rng.uniform(-0.2, 0.5, self.count)
            self.vel_x = rng.uniform(-0.8, 0.8, self.count)
        elif self.ptype == "data":
            self.chars = [chr(0x30A0 + i % 96) for i in range(self.count)]
        else:
            self.vel_y = 0
            self.vel_x = 0

    def update(self, dt: float):
        """Update particle positions."""
        self.py += self.vel_y * dt * 60
        self.px += self.vel_x * dt * 60
        self.phase += dt * 2

        # Wrap around
        self.py = np.mod(self.py, self.h)
        self.px = np.mod(self.px, self.w)

    def draw(self, canvas: np.ndarray, t: float) -> np.ndarray:
        """Draw particles onto canvas."""
        out = canvas.copy()
        
        if self.ptype == "data":
            for i in range(self.count):
                py = np.clip(self.py[i], 0, self.h - 1)
                px = np.clip(self.px[i], 0, self.w - 1)
                y, x = int(py), int(px)
                val = int(50 + 100 * np.sin(self.phase[i]))
                out[y, x] = np.minimum(out[y, x].astype(np.float32) + val, 255)
        else:
            for i in range(self.count):
                py = np.clip(self.py[i], 0, self.h - 1)
                px = np.clip(self.px[i], 0, self.w - 1)
                y, x = int(py), int(px)
                if 0 <= x < self.w and 0 <= y < self.h:
                    twinkle = 0.7 + 0.3 * np.sin(self.phase[i])
                    alpha = self.alpha[i] * twinkle
                    size = self.size[i]
                    
                    # Draw glow - simplified
                    if size > 2.0:
                        radius = int(size * 2)
                        y0 = max(0, y - radius)
                        y1 = min(self.h, y + radius + 1)
                        x0 = max(0, x - radius)
                        x1 = min(self.w, x + radius + 1)
                        grad_h = y1 - y0
                        grad_w = x1 - x0
                        if grad_h > 0 and grad_w > 0:
                            grad = np.exp(-np.linspace(0, 1, min(grad_h, grad_w)) / 0.6)
                            grad = grad[:, None, None] * 80
                            out[y0:y1, x0:x1] = np.minimum(
                                out[y0:y1, x0:x1].astype(np.float32) + grad[:grad_h, :grad_w],
                                255
                            )
                    
                    # Draw core
                    out[y, x] = np.minimum(
                        out[y, x].astype(np.float32) + 150 * alpha,
                        255
                    )
        
        return out


def _get_palette(theme: str, w: int, h: int, seed: int) -> List[Tuple[int, int, int]]:
    """Get palette for a theme."""
    config = _THEME_CONFIGS.get(theme, _THEME_CONFIGS["aurora"])
    palette = config.palette.copy()
    
    rng = np.random.default_rng(seed)
    for i in range(len(palette)):
        r = int(rng.uniform(0.8, 1.2) * palette[i][0])
        g = int(rng.uniform(0.8, 1.2) * palette[i][1])
        b = int(rng.uniform(0.8, 1.2) * palette[i][2])
        palette[i] = (np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255))
    
    return palette


def _create_background(w: int, h: int, theme: str, seed: int, t: float) -> np.ndarray:
    """Create base background for theme."""
    palette = _get_palette(theme, w, h, seed)
    
    # Create gradient background
    bg = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        progress = y / h
        t_progress = t * 0.5
        # Interpolate between two palette colors based on position
        idx1 = int(progress * len(palette)) % len(palette)
        idx2 = int((progress + 0.3) % len(palette)) % len(palette)
        mix = (math.sin(t_progress + progress * math.pi) + 1) / 2
        
        c1 = palette[idx1]
        c2 = palette[idx2]
        r = int(c1[0] * (1 - mix) + c2[0] * mix)
        g = int(c1[1] * (1 - mix) + c2[1] * mix)
        b = int(c1[2] * (1 - mix) + c2[2] * mix)
        
        bg[y, :] = (r, g, b)
    
    # Add some noise for texture
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 10, (h, w, 3), dtype=np.uint8)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return bg


def _create_mountain_layer(w: int, h: int, seed: int, offset_y: float, 
                           color: Tuple[int, int, int], thickness: float) -> np.ndarray:
    """Create a mountain layer."""
    layer = np.zeros((h, w, 3), np.uint8)
    rng = np.random.default_rng(seed)
    
    num_peaks = 5
    for i in range(num_peaks):
        base_x = int(rng.uniform(0, w))
        base_y = int(h * (0.6 + 0.2 * i / num_peaks) + offset_y)
        peak_width = int(w * 0.15)
        peak_height = int(h * 0.25)
        peak_y = max(0, min(h, base_y - peak_height // 2))
        peak_y_top = min(h, peak_y + peak_height)
        
        for y in range(peak_y, peak_y_top):
            progress = (y - peak_y) / peak_height if peak_height > 0 else 0
            wave = math.sin(progress * math.pi * 2 + i) * 0.2
            width = int(peak_width * (1 - progress) * (1 + wave))
            x_center = int(base_x + wave * w * 0.1)
            
            x_left = max(0, x_center - width // 2)
            x_right = min(w, x_center + width // 2)
            color_scaled = (
                int(color[0] * (1 - progress * 0.5)),
                int(color[1] * (1 - progress * 0.5)),
                int(color[2] * (1 - progress * 0.5)),
            )
            
            for x in range(x_left, x_right):
                layer[y, x] = color_scaled
    
    return layer


def _draw_gradient(canvas: np.ndarray, p1: Tuple[float, float], p2: Tuple[float, float], 
                  color: Tuple[int, int, int], alpha: float = 1.0) -> np.ndarray:
    """Draw a gradient between two points."""
    canvas = canvas.astype(np.float32)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx*dx + dy*dy)
    
    if length < 1e-6:
        canvas[p1[1], p1[0]] = (canvas[p1[1], p1[0]] * (1-alpha) + np.array(color) * alpha).astype(np.float32)
        return canvas
    
    nx, ny = -dy / length, dx / length
    c = np.array(color, dtype=np.float32)
    # Ensure the blend colour carries an alpha channel to match 4ch canvases.
    if c.ndim == 1 and len(c) == 3:
        c = np.concatenate([c, [255.0]])
    c = c[:4]
    
    for i in range(int(length)):
        t = i / length
        x = int(np.clip(p1[0] + dx * t, 0, canvas.shape[1] - 1))
        y = int(np.clip(p1[1] + dy * t, 0, canvas.shape[0] - 1))
        
        px, py = x - p1[0], y - p1[1]
        dist = abs(px * nx + py * ny)
        
        if dist < 1.0:
            fade = 1.0 - dist
            canvas[y, x] = (canvas[y, x] * (1-alpha*fade) + c * alpha * fade).astype(np.float32)
    
    return canvas


def _create_celestial(w: int, h: int, seed: int, t: float) -> np.ndarray:
    """Create celestial body (sun/moon) with glow."""
    celestial = np.zeros((h, w, 4), np.float32)  # RGB + alpha
    rng = np.random.default_rng(seed)
    
    # Position based on time
    cx = w * 0.7 + math.sin(t * 0.1) * w * 0.1
    cy = h * 0.3 + math.cos(t * 0.08) * h * 0.05
    radius = int(w * 0.08)
    
    # Create gradient glow
    for r in range(radius, 0, -1):
        alpha = (1 - r / radius) * 0.3
        glow = np.zeros((h, w, 4), np.float32)
        
        # Elliptical glow
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + ((y - cy) * 0.6)**2)
        mask = dist <= r
        
        if mask.any():
            # Warm yellow/white gradient
            glow[mask, 0] = np.minimum(glow[mask, 0] + 255 * alpha, 255)
            glow[mask, 1] = np.minimum(glow[mask, 1] + 240 * alpha, 255)
            glow[mask, 2] = np.minimum(glow[mask, 2] + 200 * alpha, 255)
            glow[mask, 3] = alpha
            
            # Add to celestial layer
            mask_float = mask.astype(np.float32)[:, :, None]
            celestial = np.maximum(celestial, glow * mask_float)
    
    # Add flare
    if rng.random() > 0.5:
        flare_x = cx + radius + 5
        flare_y = cy
        flare_len = radius * 2.5
        celestial = _draw_gradient(celestial, (flare_x, flare_y), (flare_x + flare_len, flare_y), (255, 255, 220, 150), alpha=1.0)
    
    return celestial


class ProceduralScene:
    """Main scene renderer with all effects."""

    def __init__(self, req: GenerationRequest):
        self.req = req
        self.theme = req.extras.get("style", "cinematic") or "aurora"
        self.camera_motion = req.extras.get("motion", "orbit") or "orbit"
        self.camera_time = 0.0
        self.seed = req.seed or 2718
        
        # Get theme config
        self.config = _THEME_CONFIGS.get(self.theme, _THEME_CONFIGS["aurora"])
        
        # Initialize
        self.w, self.h = req.width, req.height
        self.palette = _get_palette(self.theme, self.w, self.h, self.seed)
        self.camera_offset = np.array([0.0, 0.0])
        
        # Create layers
        self.background = _create_background(self.w, self.h, self.theme, self.seed, 0.0)
        self.mountains = []
        self.particles = None
        self.celestial = None
        self._init_layers()

    def _init_layers(self):
        """Initialize scene layers."""
        # Mountains layers
        num_layers = 4
        base_colors = [(30, 30, 40), (50, 50, 60), (80, 80, 90), (120, 120, 130)]
        
        for i in range(num_layers):
            offset = i * 50
            thickness = 0.8 - i * 0.15
            color = base_colors[i] if i < len(base_colors) else (150, 150, 160)
            layer = _create_mountain_layer(self.w, self.h, self.seed + i, offset, color, thickness)
            self.mountains.append(layer)
        
        # Particles
        if self.config.has_particles:
            self.particles = ParticleSystem(
                self.w, self.h,
                count=100 if self.config.particle_type == "data" else 50,
                ptype=self.config.particle_type,
                seed=self.seed + 1000,
            )
        
        # Celestial body
        if self.config.has_celestial:
            self.celestial = _create_celestial(self.w, self.h, self.seed, 0.0)

    def update(self, dt: float):
        """Update scene state."""
        self.camera_time += dt
        
        # Update camera motion
        self.camera_offset = np.array(_camera_moves.get(self.camera_motion, lambda t, w, h: (0, 0))(
            self.camera_time, self.w, self.h
        ))
        
        # Update particles
        if self.particles:
            self.particles.update(dt)

    def render(self, t: float) -> np.ndarray:
        """Render frame at time t."""
        # Start with background
        img = self.background.copy()
        
        # Apply camera motion offset
        if np.any(self.camera_offset):
            offset_x, offset_y = int(self.camera_offset[0]), int(self.camera_offset[1])
            # Simple translation effect
            img = np.roll(img, offset_x, axis=1)
            img = np.roll(img, offset_y, axis=0)
            # Fill edges
            if abs(offset_x) > 0:
                if offset_x > 0:
                    img[:, :offset_x] = img[:, offset_x:offset_x+1].mean(axis=1, keepdims=True)
                else:
                    img[:, offset_x:] = img[:, offset_x-1:offset_x].mean(axis=1, keepdims=True)
            if abs(offset_y) > 0:
                if offset_y > 0:
                    img[:offset_y] = img[offset_y:offset_y+1].mean(axis=0, keepdims=True)
                else:
                    img[offset_y:] = img[offset_y-1:offset_y].mean(axis=0, keepdims=True)
        
        # Add mountains
        for m in self.mountains:
            img = cv2.addWeighted(img, 1.0, m, 0.6, 0)
        
        # Add celestial body
        if self.celestial is not None:
            celestial_rgb = np.clip(self.celestial[:, :, :3] * 255, 0, 255).astype(np.uint8)
            celestial_alpha = self.celestial[:, :, 3][:, :, None] / 255.0
            img = img.astype(np.float32) * (1 - celestial_alpha) + celestial_rgb.astype(np.float32) * celestial_alpha
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Add particles
        if self.particles:
            img = self.particles.draw(img, t)
        
        return img


# ============================================================================
# Urban scene engine - city skylines, streets and walking silhouettes.
# ============================================================================
def _draw_v(w: int, h: int, theme: str) -> np.ndarray:
    """A vertical night-sky gradient fitted to the theme's palette."""
    if theme in ("city", "neon", "cyber"):
        top, bot = (24, 30, 52), (92, 84, 118)
    elif theme in ("apocalypse", "zombie", "horror"):
        top, bot = (18, 20, 30), (88, 70, 64)
    else:
        top, bot = (22, 26, 48), (80, 72, 100)
    ys = np.linspace(0, 1, h)[:, None, None]
    grad = np.array(top, np.float32) * (1 - ys) + np.array(bot, np.float32) * ys
    img = np.clip(grad, 0, 255).astype(np.uint8).repeat(w, axis=1)
    # A soft glow rising from the horizon so building silhouettes read.
    glow_h = max(6, int(h * 0.20))
    y0 = max(0, h - glow_h)
    fade = np.linspace(0.0, 1.0, glow_h)[:, None, None]
    if theme in ("apocalypse", "zombie", "horror"):
        glow_col = (178, 96, 74)
    elif theme in ("city", "neon", "cyber"):
        glow_col = (130, 96, 196)
    else:
        glow_col = (150, 120, 180)
    img[y0:] = (img[y0:].astype(np.float32) * (1 - fade)
                + np.array(glow_col, np.float32) * fade).clip(0, 255).astype(np.uint8)
    return img


def _draw_zombie(img, cx, ground_y, hgt, phase, lean, seed):
    """Draw one shambling silhouette figure with an alternating gait."""
    cx = int(cx); ground_y = int(ground_y)
    body_len = max(3, int(hgt * 0.5))
    head_r = max(2, int(hgt * 0.14))
    col = (14, 13, 13)   # near-black, stands out against the lit asphalt
    hy = ground_y - hgt
    # head
    cv2.circle(img, (cx, hy), head_r, col, -1)
    # torso leaning forward
    lean_px = int(lean * hgt * 0.5)
    tx1, ty1 = cx, hy + head_r
    tx2, ty2 = cx + lean_px, hy + head_r + body_len
    cv2.line(img, (tx1, ty1), (tx2, ty2), col, max(2, hgt // 7))
    # arms reaching forward (zombie lunge)
    for arm in (-1, 1):
        cv2.line(img, (tx1 + arm * head_r // 2, ty1 + body_len // 3),
                 (tx2 + arm * hgt // 4, ty1 + body_len // 3 - hgt // 7), col, 2)
    # legs - alternating shuffle stride
    swing = int(math.sin(phase) * hgt // 7)
    cv2.line(img, (tx2, ty2), (tx2 + swing, ground_y), col, max(2, hgt // 8))
    cv2.line(img, (tx2, ty2), (tx2 - swing, ground_y), col, max(2, hgt // 8))
    return img


class CityScene:
    """Procedural urban renderer: a night city skyline with a street and a
    crowd of shuffling silhouette figures (zombies). Purely synthetic numpy +
    cv2 drawing, so it needs no weights and no GPU."""

    def __init__(self, req: GenerationRequest, theme: str):
        self.req = req
        self.w, self.h = req.width, req.height
        self.theme = theme if theme in _URBAN_THEMES else "city"
        self.camera_motion = req.extras.get("motion", "pan_right") or "pan_right"
        self.seed = req.seed or 2718
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        self.horizon = int(self.h * 0.62)
        self.camera_time = 0.0
        self.frame = 0
        self.buildings = self._make_skyline(rng)
        self.crowd = self._make_crowd(rng)

    @property
    def config(self):
        return _THEME_CONFIGS.get(self.theme)

    @property
    def look(self):
        return (self.config.look if self.config else "noir")

    # ---- construction -------------------------------------------------
    def _make_skyline(self, rng) -> list:
        buildings = []
        x = -int(self.w * 0.12)
        while x < self.w:
            bw = int(rng.uniform(self.w * 0.08, self.w * 0.18))
            bh = int(rng.uniform(self.h * 0.14, self.h * 0.46))
            r = int(rng.uniform(0, 18))
            lit = set()
            if rng.random() > 0.35:
                rows_ = max(2, bh // 18)
                cols_ = max(2, bw // 12)
                for rr in range(rows_):
                    for cc in range(cols_):
                        if rng.random() > 0.72:
                            lit.add((rr, cc))
            pal = _THEME_CONFIGS.get(self.theme, _THEME_CONFIGS["city"]).palette
            buildings.append({
                "x": x, "w": bw, "h": bh, "base": self.horizon, "r": r,
                "edge": tuple(int(v) for v in pal[3]), "lit": lit,
            })
            x += bw + int(rng.uniform(self.w * 0.04, self.w * 0.10))
        return buildings

    def _make_crowd(self, rng) -> list:
        return [{
            "x": float(rng.uniform(-20, self.w + 20)),
            "speed": float(rng.uniform(12.0, 30.0)) * (1 if rng.random() > 0.5 else -1),
            "scale": float(rng.uniform(0.35, 1.0)),
            "phase": float(rng.uniform(0, 6.28)),
            "lean": float(rng.uniform(-0.25, 0.25)),
        } for _ in range(int(rng.integers(6, 12)))]

    def update(self, dt: float):
        self.camera_time += dt
        self.frame += 1
        for c in self.crowd:
            c["x"] += c["speed"] * dt
            if c["x"] < -40:
                c["x"] = self.w + 40
            elif c["x"] > self.w + 40:
                c["x"] = -40

    # ---- render --------------------------------------------------------
    def render(self, t: float) -> np.ndarray:
        img = _draw_v(self.w, self.h, self.theme)
        img = self._draw_skyline(img)
        img = self._draw_street(img)
        img = self._draw_crowd(img, t)
        fog_col = (70, 62, 60) if self.theme in ("apocalypse", "zombie", "horror") else (78, 74, 92)
        fog_h = max(3, int(self.h * 0.10))
        y0 = max(0, self.horizon - fog_h // 2)
        fade = np.linspace(0.35, 0.0, fog_h)[:, None, None]
        img[y0:y0 + fog_h] = (
            img[y0:y0 + fog_h].astype(np.float32) * (1 - fade)
            + np.array(fog_col, np.float32) * fade
        ).astype(np.uint8)
        # A mild teal-orange grade: keeps the street bright and the figures
        # readable while still giving a cold, cinematic night mood.
        from ..effects import FilmLook
        urban_look = FilmLook(
            "urban", contrast=1.08, vignette=0.16, grain=0.012, temp=-0.02,
            shadows_shift=(0.01, 0.0, 0.03), highlights_shift=(0.02, 0.0, -0.02),
            bloom=0.06,
        )
        img = grade(img, urban_look)
        return img

    def _draw_skyline(self, img):
        for b in sorted(self.buildings, key=lambda b: -b["h"]):
            x0, x1 = max(0, b["x"]), min(self.w, b["x"] + b["w"])
            y0, y1 = max(0, b["base"] - b["h"]), b["base"]
            if x1 <= x0:
                continue
            edge = np.array(b["edge"], np.uint8)
            img[y0:y1, x0:x1] = np.ones((y1 - y0, x1 - x0, 3), np.uint8) * edge
            img[y0:y0 + 3, x0:x1] = np.clip(
                img[y0:y0 + 3, x0:x1].astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
            if (x0 + b["r"]) % 7 == 0:
                cv2.line(img, (x0 + b["w"] // 2, y0), (x0 + b["w"] // 2, y0 - 6), b["edge"], 2)
            for (r_, c_) in b["lit"]:
                px = x0 + 4 + c_ * 12
                py = y0 + 6 + r_ * 18
                if x0 + 4 <= px < x1 and 0 <= py < y1:
                    img[py:py + 3, px:px + 4] = (255, 220, 160)
        return img

    def _draw_street(self, img):
        h = self.h - self.horizon
        # Asphalt fades from a lit far sidewalk to a brighter, moonlit road
        # near the camera, so dark silhouettes read cleanly.
        y = np.linspace(0, 1, h)[:, None, None]
        top = np.array((84, 82, 92), np.float32)
        bot = np.array((138, 134, 148), np.float32)
        asphalt = (top * (1 - y) + bot * y).clip(0, 255).astype(np.uint8)
        img[self.horizon:] = asphalt.repeat(img.shape[1], 1)
        cv2.line(img, (0, self.h - 6), (self.w, self.h - 6), (150, 146, 156), 3)
        cv2.line(img, (0, self.h - round(self.h * 0.16)), (self.w, self.h - round(self.h * 0.16)),
                 (96, 94, 104), 1)
        for i in range(-20, self.w, 56):
            cv2.line(img, (i, self.h - round(self.h * 0.30)),
                     (i + 28, self.h - round(self.h * 0.24)), (118, 116, 108), 2)
        return img

    def _draw_crowd(self, img, t):
        for c in self.crowd:
            scale = c["scale"]
            ground_y = self.horizon + int((self.h - self.horizon) * scale * 0.82)
            hgt = int(max(10, (self.h - self.horizon) * scale * 0.7))
            _draw_zombie(img, c["x"], ground_y, hgt, c["phase"], c["lean"], self.seed)
        return img


class CinematicBackend(GeneratorBackend):
    """Procedural cinematic video generation backend."""
    name = "cinematic"

    def check(self) -> bool:
        return True

    def generate(self, req: GenerationRequest) -> GeneratedClip:
        """Generate video using procedural rendering.

        The theme is auto-resolved from the prompt (or an explicit ``style``),
        so free text like "zombies walking on a city street" maps to an urban
        apocalypse scene instead of silently falling back to aurora.
        """
        theme = resolve_theme(req.prompt, req.extras.get("style"))
        if theme not in _THEME_CONFIGS:
            theme = "moody"
        motion = req.extras.get("motion", "orbit") or "orbit"

        # Make sure the procedural scene picks up the resolved theme.
        req.extras = dict(req.extras)
        req.extras["style"] = theme

        urban = theme in _URBAN_THEMES
        if urban:
            scene = CityScene(req, theme)
        else:
            scene = ProceduralScene(req)
            scene.camera_motion = motion

        # Default the grade to the theme's signature look when none given.
        look = req.extras.get("look")
        if not look:
            look = getattr(scene, "look", "argo")
        # Urban scenes apply their own dedicated grade inside render(); they
        # must not be graded a second time here or the mids get crushed.
        outer_grade = None if urban else (look,)

        print(f"[cinematic] prompt -> theme '{theme}' "
              f"({'urban' if urban else 'procedural'} scene, {motion} move)")

        # Render frames
        fps = float(req.fps)
        duration = float(req.duration)
        n_frames = int(round(duration * fps))
        frames = []

        print(f"Generating {n_frames} frames at {fps} fps...")

        for t_idx in range(n_frames):
            t = t_idx / fps
            scene.update(1.0 / fps)
            frame = scene.render(t)

            # Procedural scenes need the generic grade applied here.
            if outer_grade is not None:
                frame = grade(frame, lookup(outer_grade[0]))

            frames.append(frame)

            if (t_idx + 1) % 10 == 0:
                print(f"  Rendered {t_idx + 1}/{n_frames} frames")

        clip = GeneratedClip(
            prompt=req.prompt,
            backend=self.name,
            frames=np.stack(frames),
            fps=fps,
            metadata={
                "theme": theme,
                "motion": motion,
                "look": look,
                "seed": req.seed or scene.seed,
            }
        )

        print(f"  Complete! Generated {clip.T} frames ({clip.T / clip.fps}s)")
        return clip
