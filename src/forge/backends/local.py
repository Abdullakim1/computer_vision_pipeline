"""Local latent-diffusion backend — Wan 2.1 1.3B (realistic text-to-video).

Runs the open-source Wan 2.1 T2V 1.3B model in-process on a CUDA GPU. Uses
8-bit quantization (bitsandbytes) for the text encoder so the full pipeline
fits in ~8–10 GB VRAM — runs on a free Colab T4 or any 12GB+ consumer GPU.

Falls back gracefully if torch/diffusers/bitsandbytes or a CUDA device are
not present, so the studio can always switch to ``cinematic`` or a cloud
backend.

Set ``WAN_MODEL_CACHE`` in the env to reuse a local weights folder (point
at a Hugging Face snapshot dir or a ``models/Wan2.1-T2V-1.3B-Diffusers``
checkout) — otherwise weights are streamed and cached by the Hub.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..types import GeneratedClip, GenerationRequest
from .base import GeneratorBackend


# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------
def _available():
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        import bitsandbytes  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def _snap(v: int, lo: int = 64, multiple: int = 16) -> int:
    """Snap a pixel dimension onto the grid Wan's VAE was trained at."""
    return max(lo, (int(v) // multiple) * multiple)


_REALISM_SUFFIX = (
    "photorealistic live-action film footage, cinematic lighting, "
    "shot on digital cinema camera, natural motion"
)

_DEFAULT_NEGATIVE = (
    "cartoon, anime, illustration, painting, drawing, CGI, 3d render, "
    "plastic skin, blurry, low quality, worst quality, jpeg artifacts, deformed"
)


# ---------------------------------------------------------------------------
# Model singleton — lazy-loaded once per process, thread-safe.
# ---------------------------------------------------------------------------
@dataclass
class _LoadedPipe:
    pipe: object  # diffusers.WanPipeline
    model_id: str
    loaded_at: float


_PIPE: Optional[_LoadedPipe] = None
_I2V_PIPE: Optional[_LoadedPipe] = None
_PIPE_LOCK = threading.Lock()


def _load_pipe(model_id: str, local_dir: Optional[str]):
    """Load (or reuse) the Wan 2.1 pipeline onto the GPU."""
    global _PIPE
    with _PIPE_LOCK:
        if _PIPE is not None:
            return _PIPE.pipe

        import time
        import torch
        from diffusers import WanPipeline, AutoencoderKLWan
        from transformers import AutoTokenizer, UMT5EncoderModel, BitsAndBytesConfig

        source = local_dir if local_dir and os.path.isdir(local_dir) else model_id
        print(f"[local] loading Wan pipeline from: {source}")
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(source, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            source, subfolder="text_encoder",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        vae = AutoencoderKLWan.from_pretrained(source, subfolder="vae", dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(
            source,
            text_encoder=text_encoder, tokenizer=tokenizer, vae=vae,
            dtype=torch.float16,
        )
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        try:
            pipe.enable_vae_tiling()
        except AttributeError:
            pass  # newer diffusers tiles the Wan VAE by default

        import gc; gc.collect()
        elapsed = round(time.time() - t0, 1)
        vram = round(torch.cuda.memory_allocated() / 1e9, 1)
        print(f"[local] Wan ready in {elapsed}s, VRAM used: {vram} GB")

        _PIPE = _LoadedPipe(pipe=pipe, model_id=model_id, loaded_at=time.time())
        return pipe


def _load_i2v_pipe(model_id: str, local_dir: Optional[str]):
    """Load (or reuse) the Wan 2.1 image-to-video pipeline onto the GPU."""
    global _I2V_PIPE
    with _PIPE_LOCK:
        if _I2V_PIPE is not None:
            return _I2V_PIPE.pipe

        import gc
        import time

        import torch
        from diffusers import WanI2VPipeline

        source = local_dir if local_dir and os.path.isdir(local_dir) else model_id
        print(f"[local] loading Wan I2V pipeline from: {source}")
        t0 = time.time()
        pipe = WanI2VPipeline.from_pretrained(source, torch_dtype=torch.float16)
        pipe.to("cuda")
        try:
            pipe.enable_vae_tiling()
        except AttributeError:
            pass  # newer diffusers tiles the Wan VAE by default

        gc.collect()
        elapsed = round(time.time() - t0, 1)
        vram = round(torch.cuda.memory_allocated() / 1e9, 1)
        print(f"[local] Wan I2V ready in {elapsed}s, VRAM used: {vram} GB")

        _I2V_PIPE = _LoadedPipe(pipe=pipe, model_id=model_id, loaded_at=time.time())
        return pipe


# ---------------------------------------------------------------------------
# Backend facade
# ---------------------------------------------------------------------------
class LocalLatentBackend(GeneratorBackend):
    """Realistic local video generation via Wan 2.1 1.3B (8-bit, CUDA)."""

    name = "local"

    DEFAULT_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    DEFAULT_I2V_MODEL = "Wan-AI/Wan2.1-I2V-1.3B-Diffusers"

    def __init__(self, **env):
        super().__init__(**env)
        self.model_id = os.getenv("WAN_MODEL_ID", self.DEFAULT_MODEL)
        self.i2v_model_id = os.getenv("WAN_I2V_MODEL_ID", self.DEFAULT_I2V_MODEL)
        self.local_dir = os.getenv("WAN_MODEL_CACHE", "models/Wan2.1-T2V-1.3B-Diffusers")
        self.i2v_local_dir = os.getenv(
            "WAN_I2V_MODEL_CACHE", "models/Wan2.1-I2V-1.3B-Diffusers"
        )

    def check(self) -> bool:
        return _available()

    def generate(self, req: GenerationRequest) -> GeneratedClip:
        if not _available():
            raise RuntimeError(
                "local latent backend unavailable: need torch + diffusers + "
                "bitsandbytes on a CUDA GPU. Use 'cinematic' for offline, or "
                "'colab' / 'luma' cloud backends."
            )
        import torch

        pipe = _load_pipe(self.model_id, self.local_dir)

        # --- prompt enhancement ------------------------------------------------
        style = req.extras.get("style", "realistic")
        prompt = req.prompt
        if style not in ("none", "raw"):
            prompt = f"{prompt}, {_REALISM_SUFFIX}" if style == "realistic" else f"{prompt}, {style} style"
        negative = req.negative_prompt or _DEFAULT_NEGATIVE

        # --- dims + frames -----------------------------------------------------
        w, h = _snap(req.width), _snap(req.height)
        fps = max(5, min(int(req.fps), 24))
        duration = max(1.0, min(float(req.duration), 10.0))
        num_frames = max(9, min(int(round(duration * fps)), 121))

        steps = int(req.extras.get("steps", 25))
        guidance = float(req.extras.get("guidance_scale", 5.0))
        seed = int(req.seed) if req.seed is not None else 2718
        generator = torch.Generator("cuda").manual_seed(seed)

        print(
            f"[local] sampling Wan2.1: {w}x{h} x {num_frames}f @ {fps}fps "
            f"({duration:.1f}s), steps={steps}, cfg={guidance}, seed={seed}"
        )

        # --- sample ------------------------------------------------------------
        import time
        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=w, height=h,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        elapsed = round(time.time() - t0, 1)
        print(f"[local] sampling finished in {elapsed}s")

        pil_frames = result.frames[0]
        frames = np.stack([np.asarray(f, dtype=np.uint8) for f in pil_frames])

        return GeneratedClip(
            prompt=req.prompt,
            backend="local",
            frames=frames,
            fps=float(fps),
            metadata={
                "model": self.model_id,
                "enhanced_prompt": prompt,
                "negative_prompt": negative,
                "seed": seed,
                "resolution": f"{w}x{h}",
                "steps": steps,
                "guidance_scale": guidance,
                "elapsed_s": elapsed,
            },
        )

    # ------------------------------------------------------------------
    def image_to_video(self, image_path, req: GenerationRequest) -> GeneratedClip:
        """Image-to-video via the Wan image-to-video model (CUDA, 8-bit)."""
        if not _available():
            raise RuntimeError(
                "local image-to-video unavailable: need torch + diffusers + "
                "bitsandbytes on a CUDA GPU. Use 'colab' / 'kling' / 'seedance' "
                "for image-to-video on remote backends."
            )
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

        import time
        from PIL import Image

        import torch

        from ..prompts import prompt_image_motion  # small helper (kept private)

        pipe = _load_i2v_pipe(self.i2v_model_id, self.i2v_local_dir)
        style = req.extras.get("style", "realistic")
        prompt = req.prompt or prompt_image_motion(image_path)
        if style not in ("none", "raw"):
            prompt = f"{prompt}, {_REALISM_SUFFIX}" if style == "realistic" else f"{prompt}, {style} style"
        negative = req.negative_prompt or _DEFAULT_NEGATIVE

        w, h = _snap(req.width), _snap(req.height)
        fps = max(5, min(int(req.fps), 24))
        duration = max(1.0, min(float(req.duration), 10.0))
        num_frames = max(9, min(int(round(duration * fps)), 121))
        steps = int(req.extras.get("steps", 25))
        guidance = float(req.extras.get("guidance_scale", 5.0))
        seed = int(req.seed) if req.seed is not None else 2718
        generator = torch.Generator("cuda").manual_seed(seed)
        image = Image.open(image_path).convert("RGB")

        print(f"[local] sampling Wan2.1 I2V: {w}x{h} x {num_frames}f @ {fps}fps, src={os.path.basename(image_path)}")
        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=image,
            width=w, height=h,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        elapsed = round(time.time() - t0, 1)
        print(f"[local2] I2V sampling finished in {elapsed}s")

        pil_frames = result.frames[0]
        frames = np.stack([np.asarray(f, dtype=np.uint8) for f in pil_frames])

        return GeneratedClip(
            prompt=prompt,
            backend="local",
            frames=frames,
            fps=float(fps),
            metadata={
                "model": self.i2v_model_id,
                "kind": "i2v",
                "source_image": os.path.basename(image_path),
                "enhanced_prompt": prompt,
                "negative_prompt": negative,
                "seed": seed,
                "resolution": f"{w}x{h}",
                "steps": steps,
                "guidance_scale": guidance,
                "elapsed_s": elapsed,
            },
        )
