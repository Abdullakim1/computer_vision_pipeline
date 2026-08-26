"""CineForge studio - Advanced Video Generation Studio.

CineForge ties backends, conditioning, motion, effects, prompts, and
metrics into a high-level creative API:

    studio.text_to_video("aurora over a valley")  -> GeneratedClip
    studio.image_to_video("photo.jpg")            -> GeneratedClip
    studio.direct("a sun sets over ancient ruins") -> a short film
    studio.upsample(clip, factor=2)               -> motion interpolation
    studio.regrade(clip, "golden")                -> color grade pass
    studio.batch_generate(prompts, n=3)            -> list of clips
    studio.generate_storyboard(story, beats=5)    -> movie
    studio.create_compilation(clips, transition='crossfade') -> final video
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import metrics
from .backends import create_backend
from .conditioning import SemanticGuide
from .effects import grade, lookup
from .motion import interpolate_clip, ken_burns
from .prompts import CinematicPrompter
from .types import GeneratedClip, GenerationRequest


class CineForgeStudio:
    """Advanced video generation studio with multiple backends and tools."""

    def __init__(self, config=None, seed=None):
        self.config = dict(config or {})
        self.seed = seed or self.config.get("seed", 2718)
        self.prompter = CinematicPrompter(self.seed)
        self.guide = SemanticGuide()
        self._weights = {}
        self._generation_count = 0

        # ------------------------------------------------------------------
    @staticmethod
    def backends():
        """List available backends and their readiness."""
        info = []
        for name in ("cinematic", "local", "colab", "kaggle", "luma", "seedance", "kling", "veo"):
            try:
                b = create_backend(name)
                ready = b.check()
                entry = {"name": name, "ready": ready,
                         "desc": (type(b).__doc__ or "").strip()[:100]}
                # Surface the "busy" flag from remote backends (colab/kaggle)
                # so the UI can warn the user that the server is up but
                # currently running another generation.
                entry["busy"] = getattr(b, "busy", False)
                info.append(entry)
            except Exception as exc:
                info.append({"name": name, "ready": False,
                             "desc": str(exc)[:100], "busy": False})
        return info

    # ------------------------------------------------------------------
    def text_to_video(self, prompt, backend="cinematic", width=960, height=540,
                      fps=24, duration=4.0, seed=None, negative_prompt=None, **kw):
        """Generate a clip from a text prompt using the chosen backend."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt is required and must be non-empty")
        backend_obj = create_backend(backend)
        if not backend_obj.check():
            ready = [x["name"] for x in self.backends() if x["ready"]]
            raise RuntimeError(f"backend {backend!r} not ready; available: {ready}")
        req = GenerationRequest(
            prompt=prompt.strip(),
            negative_prompt=negative_prompt or "",
            width=int(width), height=int(height), fps=int(fps),
            duration=float(duration),
            seed=seed if seed is not None else (self.seed or hash(prompt) & 0xffffffff),
            extras=kw,
        )
        clip = backend_obj.generate(req)
        self._generation_count += 1
        clip.metadata["generation_id"] = self._generation_count
        clip.metadata["metrics"] = metrics.analyze(clip.frames, float(req.fps))
        return clip

        # ------------------------------------------------------------------
    def image_to_video(self, image_path, backend=None, prompt=None,
                       width=960, height=540, fps=24, duration=4.0,
                       look="argo", interp=1, **kw):
        """Animate a still image.

        With a diffusion ``backend`` (``colab``/``kaggle``/``local``/``kling``/
        ``seedance``) this dispatches the real image-to-video pipeline (Wan I2V
        on Colab A100 / local GPU, or the Seedance/Kling image-to-video APIs).
        Note: free Kaggle GPUs (T4/P100) are text-to-video only. With ``cinematic``
        (the default, GPU-free) it runs a Ken-Burns pan + motion-interpolation
        pass instead.
        """
        import os

        backend = backend or os.getenv("default_backend", "cinematic")
        image_path = str(image_path)

        # --- real diffusion backends ----------------------------------------
        if backend in ("colab", "kaggle", "local"):
            obj = create_backend(backend)
            if not obj.check():
                raise RuntimeError(f"backend {backend!r} is not ready")
            req = GenerationRequest(
                prompt=prompt or "",
                negative_prompt=kw.get("negative_prompt", ""),
                width=int(width), height=int(height), fps=int(fps),
                duration=float(duration),
                seed=kw.get("seed") if kw.get("seed") is not None else None,
                motion_strength=float(kw.get("motion_strength", 0.6)),
                extras={
                    "style": kw.get("style", "realistic"),
                    "motion": kw.get("motion"),
                    "steps": kw.get("steps"),
                    "guidance_scale": kw.get("guidance_scale"),
                },
            )
            clip = obj.image_to_video(image_path, req)
            self._generation_count += 1
            clip.metadata["metrics"] = metrics.analyze(clip.frames, float(req.fps))
            clip.metadata["backend"] = backend
            return clip

        if backend in ("kling", "seedance"):
            obj = create_backend(backend)
            if not obj.check():
                raise RuntimeError(f"backend {backend!r} is not ready")
            req = GenerationRequest(
                prompt=prompt or "",
                negative_prompt=kw.get("negative_prompt", ""),
                width=int(width), height=int(height), fps=int(fps),
                duration=float(duration),
                seed=kw.get("seed") if kw.get("seed") is not None else None,
                # these backends read ``first_frame`` as a base64 image path
                first_frame=image_path,
                extras={
                    "style": kw.get("style", "cinematic"),
                    "motion": kw.get("motion", "camera_orbit"),
                    "motion_strength": kw.get("motion_strength", 0.6),
                },
            )
            clip = obj.generate(req)
            self._generation_count += 1
            clip.metadata["metrics"] = metrics.analyze(clip.frames, float(req.fps))
            clip.metadata["backend"] = backend
            clip.metadata["source_image"] = os.path.basename(image_path)
            return clip

        if backend == "veo":
            raise RuntimeError(
                "veo (Gemini) backend is text-to-video only — it has no "
                "image-to-video support. Use 'colab', 'kaggle', 'local', "
                "'kling' or 'seedance' for image-to-video."
            )

        # --- cinematic fallback: GPU-free Ken Burns -------------------------
        import cv2
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(image_path)
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames = ken_burns(img_rgb, w=width, h=height,
                           dur=float(duration), fps=int(fps))
        if interp and interp > 1:
            bgr_frames = [f[:, :, ::-1] for f in frames]
            frames = [f[:, :, ::-1] for f in interpolate_clip(bgr_frames, interp)]
        frames = [grade(f, lookup(look)) for f in frames]
        fps_out = float(fps) * (interp or 1)
        clip = GeneratedClip(f"image-to-video: {Path(image_path).name}", "cinematic",
                             np.stack(frames), fps_out,
                             metadata={"source": image_path, "look": look})
        clip.metadata["metrics"] = metrics.analyze(clip.frames, fps_out)
        clip.metadata["backend"] = backend
        self._generation_count += 1
        return clip

    # ------------------------------------------------------------------
    def direct(self, story, beats=3, backend="cinematic", width=960, height=540,
               fps=24, fade=6, seed=None, **kw):
        """Storyboard a story into a short film with crossfaded shots."""
        shots = self.prompter.script(story, beats=beats, seed=seed)
        clips = []
        for i, shot in enumerate(shots):
            c = self.text_to_video(
                shot["prompt"], backend=backend, width=width,
                height=height, fps=fps, duration=2.2,
                seed=shot["seed"], negative_prompt=shot["negative_prompt"],
            )
            self._weights[i] = shot
            clips.append(c)
        merged = _crossfade([c.frames for c in clips], fade)
        rect = _pad_to(merged, width, height)
        combined = GeneratedClip(story, backend, np.stack(rect),
                                 float(fps), metadata={"shots": shots})
        combined.metadata["metrics"] = metrics.analyze(combined.frames, float(fps))
        self._generation_count += 1
        return combined

    # ------------------------------------------------------------------
    @staticmethod
    def load_video(path, max_frames=600):
        """Load an existing video file into a GeneratedClip (RGB frames)."""
        import cv2
        path = str(path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frames = []
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise ValueError(f"no frames decoded from {path}")
        clip = GeneratedClip(Path(path).name, "file", np.stack(frames),
                             float(fps), metadata={"source": path})
        clip.metadata["metrics"] = metrics.analyze(clip.frames, float(fps))
        return clip

    # ------------------------------------------------------------------
    def remix(self, clip, target_prompt=None, keep=0.6, seed=None):
        """Re-select frames semantically guided toward a target prompt."""
        frames = [np.asarray(f) for f in clip.frames]
        ranked = self.guide.rank_frames(frames)
        keep_n = max(1, int(len(frames) * keep))
        pick = sorted(i for i, _ in ranked[:keep_n])
        title = target_prompt or clip.prompt
        out = GeneratedClip(title, clip.backend, np.stack(frames[pick]),
                            float(clip.fps),
                            metadata={"remixed": True, "kept": len(pick)})
        out.metadata["metrics"] = metrics.analyze(out.frames, float(clip.fps))
        self._generation_count += 1
        return out

    # ------------------------------------------------------------------
    def batch_generate(self, prompts, backend="cinematic", width=960, height=540,
                       fps=24, duration=4.0, n_per_prompt=1, seed=None, **kw):
        """Generate multiple videos from multiple prompts."""
        results = []
        for prompt_list in prompts:
            for i in range(n_per_prompt):
                prompt = prompt_list if isinstance(prompt_list, str) else prompt_list[i]
                clip = self.text_to_video(
                    prompt, backend=backend, width=width, height=height,
                    fps=fps, duration=duration, seed=seed,
                    **kw
                )
                results.append(clip)
        return results

    # ------------------------------------------------------------------
    def generate_storyboard(self, story, beats=5, style="cinematic", **kw):
        """Generate a complete short film storyboard."""
        return self.direct(story, beats=beats, **kw)

    # ------------------------------------------------------------------
    def create_compilation(self, clips, transition='crossfade', duration=3.0):
        """Create a compilation video from multiple clips with transitions."""
        if not clips:
            raise ValueError("No clips provided for compilation")
        
        # Calculate total duration
        total_duration = sum(c.duration_s for c in clips) + len(clips) * duration
        max_duration = 30.0  # Cap at 30 seconds
        
        # Adjust clip durations if needed
        adjusted_clips = []
        remaining_duration = min(total_duration, max_duration)
        
        for clip in clips:
            if remaining_duration <= 0:
                break
            clip_duration = min(clip.duration_s, remaining_duration)
            adjusted_clips.append((clip, clip_duration))
            remaining_duration -= clip_duration + duration
        
        # Build transitions
        all_frames = []
        for i, (clip, clip_dur) in enumerate(adjusted_clips):
            start_idx = int(i * duration * clip.fps)
            end_idx = min(start_idx + int(clip_dur * clip.fps), clip.T)
            frames = clip.frames[start_idx:end_idx]
            all_frames.append(frames)
        
        # Crossfade implementation
        if transition == 'crossfade':
            merged = _crossfade(all_frames, fade=6)
        elif transition == 'fade':
            merged = _simple_fade(all_frames, fade=12)
        else:
            merged = all_frames[0]
        
        # Final clip
        combined = GeneratedClip(
            f"Compilation ({transition}", "cinematic",
            np.stack(merged), 24.0,
            metadata={"clips": len(adjusted_clips), "transition": transition}
        )
        self._generation_count += 1
        return combined

    # ------------------------------------------------------------------
    def generate_prompt_variations(self, base_prompt, n=3, **kw):
        """Generate multiple prompt variations for A/B testing."""
        variations = []
        for i in range(n):
            if i == 0:
                variations.append(base_prompt)
            else:
                modifier = f" cinematic shot {i+1}, re-framed, {kw.get('style', 'cinematic')} style"
                variations.append(base_prompt + modifier)
        return variations

    # ------------------------------------------------------------------
    @staticmethod
    def report(clip) -> None:
        print("\u2500" * 70)
        print(f"Generated Clip")
        print(f"\u2500" * 70)
        print(f"Prompt:       {clip.prompt}")
        print(f"Backend:      {clip.backend}")
        print(f"Resolution:   {clip.width}x{clip.height}")
        print(f"FPS:          {float(clip.fps):.2f}")
        print(f"Duration:     {float(clip.duration_s):.2f}s")
        print(f"Frames:       {clip.T}")
        print(f"Seed:         {clip.metadata.get('seed', 'N/A')}")
        print(f"Metrics:")
        for k, v in clip.metadata.get("metrics", {}).items():
            print(f"  {k:>25}: {v}")

    @staticmethod
    def regrade(clip, look="golden"):
        clip.frames = np.stack([grade(f, lookup(look)) for f in clip.frames])
        return clip

    @staticmethod
    def upsample(clip, factor=2):
        if not (1 <= factor <= 8):
            raise ValueError("upsample factor must be between 1 and 8")
        bgr_frames = [f[:, :, ::-1] for f in clip.frames]
        done = [f[:, :, ::-1] for f in interpolate_clip(bgr_frames, factor)]
        return GeneratedClip(clip.prompt, clip.backend, np.stack(done),
                             float(clip.fps) * factor, metadata=dict(clip.metadata))

    @staticmethod
    def get_available_looks():
        """Get list of available color grading presets."""
        return list(lookup('').__class__.presets().keys())

    @staticmethod
    def get_available_themes():
        """Get list of available procedural themes."""
        from .backends.cinematic import _THEME_CONFIGS
        return list(_THEME_CONFIGS.keys())

    @staticmethod
    def get_available_motions():
        """Get list of available camera motions."""
        from .backends.cinematic import _camera_moves
        return list(_camera_moves.keys())

    # ------------------------------------------------------------------
    def generate_portfolio(self, outdir="media/portfolio", themes=None, **kw):
        """Generate a self-contained portfolio: clips, gifs, posters,
        a quality report and a shareable ``index.html`` gallery.

        Returns a dict with ``clips``, ``report`` and ``gallery`` keys.
        """
        from .showcase import build_portfolio
        return build_portfolio(self, outdir=outdir, themes=themes, **kw)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _crossfade(frame_lists, fade):
    out = []
    if not frame_lists:
        return out
    out.extend(frame_lists[0])
    for i in range(1, len(frame_lists)):
        prev, nxt = frame_lists[i - 1], frame_lists[i]
        f = min(fade, max(1, len(prev) // 2), max(1, len(nxt) // 2))
        a = np.asarray(prev[-f:], np.float32)
        b = np.asarray(nxt[:f], np.float32)
        ws = np.linspace(0, 1, f, dtype=np.float32)[:, None, None, None]
        blend = (a * (1 - ws) + b * ws).astype(np.uint8)
        out = out[:-f] + list(blend)
        out.extend(nxt[f:])
    return out


def _simple_fade(frame_lists, fade):
    out = []
    if not frame_lists:
        return out
    out.extend(frame_lists[0])
    for i in range(1, len(frame_lists)):
        prev, nxt = frame_lists[i - 1], frame_lists[i]
        f = min(fade, max(1, len(prev) // 2), max(1, len(nxt) // 2))
        a = np.asarray(prev[-f:], np.float32)
        b = np.asarray(nxt[:f], np.float32)
        ws = np.linspace(0, 1, f, dtype=np.float32)[:, None, None, None]
        blend = (a * (1 - ws) + b * ws).astype(np.uint8)
        out = out[:-f] + list(blend)
        out.extend(nxt[f:])
    return out


def _pad_to(frames, width, height):
    out = []
    for f in frames:
        h, w = f.shape[:2]
        if (w, h) == (width, height):
            out.append(np.asarray(f))
            continue
        canvas = np.zeros((height, width, 3), np.uint8)
        ox, oy = max(0, (width - w) // 2), max(0, (height - h) // 2)
        hh = min(h, height - oy)
        ww = min(w, width - ox)
        canvas[oy:oy + hh, ox:ox + ww] = f[:hh, :ww]
        out.append(canvas)
    return out
