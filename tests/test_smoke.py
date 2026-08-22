"""Smoke test: import the studio, generate a short clip, export artifacts."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.forge import CineForgeStudio


def test_backends_list():
    studio = CineForgeStudio()
    names = {b["name"] for b in studio.backends()}
    assert {"cinematic", "local", "luma", "seedance", "kling"} <= names
    assert any(b["ready"] for b in studio.backends())


def test_luma_backend_payload():
    """Luma backend constructs the correct Dream Machine API payload."""
    from unittest.mock import patch, MagicMock
    from src.forge.backends.luma import LumaBackend, LumaRequest
    from src.forge.types import GenerationRequest

    backend = LumaBackend()

    # _submit_task builds the correct payload for the Dream Machine API
    req = GenerationRequest(
        prompt="A serene lake surrounded by mountains at sunset",
        width=1280, height=720, fps=24, duration=5.0,
    )
    luma_req = LumaRequest(
        prompt=backend._build_prompt(req.prompt, "cinematic", "camera_orbit"),
        model="ray-2",
        resolution=backend._resolution_for(req.width, req.height),
        aspect_ratio=backend._aspect_ratio_for(req.width, req.height),
        duration=min(max(int(req.duration), 2), 10),
    )

    with patch("src.forge.backends.luma.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"id": "test-gen-id"}
        mock_post.return_value = mock_resp

        task_id = backend._submit_task(luma_req)
        assert task_id == "test-gen-id"

        sent = mock_post.call_args.kwargs["json"]
        assert sent["model"] == "ray-2"
        assert sent["duration"] == "5s"
        assert sent["resolution"] == "720p"
        assert sent["aspect_ratio"] == "16:9"
        assert "motion_preset" not in sent
        assert "image" not in sent
        assert "keyframes" not in sent

    # Polling checks the ``state`` field (not ``status``)
    with patch("src.forge.backends.luma.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "id": "gen-1", "state": "completed",
            "assets": {"video": "https://cdn.luma.test/v.mp4"},
        }
        mock_get.return_value = mock_resp

        data = backend._poll_task("gen-1")
        assert data["state"] == "completed"
        assert data["assets"]["video"] == "https://cdn.luma.test/v.mp4"


def test_generate_cinematic():
    studio = CineForgeStudio(seed=2718)
    clip = studio.text_to_video(
        "cinematic aurora over snowy mountains, orbit camera",
        backend="cinematic", width=320, height=180, fps=15, duration=2.0,
    )
    assert clip.frames.shape == (30, 180, 320, 3)
    assert clip.frames.dtype == np.uint8
    studio.report(clip)


def test_portfolio_gallery_html(tmp_path):
    """The self-contained portfolio gallery must render with CSS braces intact."""
    from src.forge.showcase import _gallery_html
    _write_dummy_assets(tmp_path, "a")
    rows = [{
        "theme": "aurora", "motion": "orbit",
        "gif": str(tmp_path / "a.gif"), "poster": str(tmp_path / "a.jpg"),
        "perceptual_quality": 0.55, "avg_sharpness": 12.5,
    }]
    out = _gallery_html(rows, tmp_path, {"themes": 1, "clips": 1,
                                         "frames": 8, "seconds": 0.5})
    html = Path(out).read_text(encoding="utf-8")
    assert "--bg:#0b0e14" in html          # CSS braces survive replacement
    assert "data:image/gif;base64," in html  # previews embedded inline
    assert "0.55" in html and "<h3>Aurora</h3>" in html


def _write_dummy_assets(tmp_path, stem):
    """Write tiny placeholder gif + jpg so base64 embedding runs for real."""
    import numpy as _np
    import cv2 as _cv2
    from PIL import Image
    img = _np.zeros((8, 16, 3), _np.uint8)
    Image.fromarray(img).save(str(tmp_path / f"{stem}.gif"))
    _cv2.imwrite(str(tmp_path / f"{stem}.jpg"), img)


if __name__ == "__main__":
    test_backends_list()
    print("backends OK")
    test_generate_cinematic()
    studio = CineForgeStudio(seed=7)
    c = studio.text_to_video("aurora over a mountain valley, teal and violet",
                             width=320, height=180, fps=15, duration=3.0)
    c.write_video("outputs/smoke_aurora.mp4")
    c.to_gif("outputs/smoke_aurora.gif", fps=8, scale=0.5)
    print("wrote outputs/smoke_aurora.*")
