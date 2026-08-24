"""Unit tests for the Colab backend wiring.

No network and no GPU are required. We mock ``requests.post``/``requests.get``
and the frame decoder so the *payload construction* (the verified logic here)
can be asserted exactly.
"""
from __future__ import annotations
import os
import tempfile
from unittest import mock

import numpy as np

from src.forge.backends import colab as colab_mod
from src.forge.backends.colab import ColabBackend
from src.forge.types import GenerationRequest


class FakeResp:
    def __init__(self, content=b"fake-mp4-bytes", status_code=200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {"X-Seed": "4242", "X-Model": "wan-14b",
                                   "X-Elapsed-S": "9.9", "X-Fps": "16"}


def _req(prompt="zombies in a city", width=832, height=480, fps=16, duration=3.0,
         steps=20, guidance_scale=6.0, seed=7, style="realistic", motion=None):
    return GenerationRequest(
        prompt=prompt, width=width, height=height, fps=fps, duration=duration,
        seed=seed,
        extras={"steps": steps, "guidance_scale": guidance_scale,
                "style": style, "motion": motion},
    )


def test_check_uses_health():
    with mock.patch.object(colab_mod.requests, "get",
                           return_value=FakeResp(status_code=200)):
        os.environ["COLAB_BASE_URL"] = "https://x.trycloudflare.com"
        b = ColabBackend()
        assert b.check() is True
    with mock.patch.object(colab_mod.requests, "get",
                           side_effect=colab_mod.requests.RequestException("no")):
        assert ColabBackend().check() is False
    os.environ.pop("COLAB_BASE_URL", None)


def test_text_to_video_posts_json_to_generate():
    os.environ["COLAB_BASE_URL"] = "https://x.trycloudflare.com"
    b = ColabBackend()
    req = _req()
    with mock.patch.object(colab_mod.requests, "post",
                           return_value=FakeResp(content=b"\x00\x00ftyp")) as mp, \
         mock.patch.object(colab_mod, "_decode_frames",
                           return_value=[np.zeros((8, 8, 3), dtype=np.uint8)] * 5):
        clip = b.generate(req)
        called_url = mp.call_args.args[0] if mp.call_args.args else mp.call_args[0][0]
    assert called_url.endswith("/generate")
    body = mp.call_args.kwargs["json"]
    assert body["prompt"].startswith("zombies in a city")
    assert "photorealistic live-action" in body["prompt"]  # realism suffix
    assert body["width"] == 832 and body["height"] == 480
    assert body["seed"] == 7
    assert clip.backend == "colab"
    assert clip.T == 5
    assert clip.metadata["kind"] == "t2v"


def test_image_to_video_posts_multipart_to_image():
    os.environ["COLAB_BASE_URL"] = "https://x.trycloudflare.com"
    b = ColabBackend()
    req = _req(prompt="a still life", width=512, height=320)
    # write a real tiny PNG image
    img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    from PIL import Image
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(img.name)
    img.close()
    try:
        with mock.patch.object(colab_mod.requests, "post",
                               return_value=FakeResp(content=b"ftypfake")) as mp, \
             mock.patch.object(colab_mod, "_decode_frames",
                               return_value=[np.zeros((4, 4, 3), dtype=np.uint8)] * 3):
            clip = b.image_to_video(img.name, req)
    finally:
        os.unlink(img.name)
    called_url = mp.call_args.args[0] if mp.call_args.args else mp.call_args[0][0]
    assert called_url.endswith("/image")
    files = mp.call_args.kwargs["files"]
    assert "file" in files
    data = mp.call_args.kwargs["data"]
    assert data["prompt"].startswith("a still life")
    assert data["width"] == 512
    assert clip.backend == "colab"
    assert clip.metadata["kind"] == "i2v"
    assert clip.metadata["source_image"] == os.path.basename(img.name)
