"""Kaggle-hosted Wan backend - realistic text-to-video on a Kaggle GPU.

The notebooks in ``kaggle/`` run the same Wan server stack as the Colab
notebook on a Kaggle GPU session (free T4/P100 tier works) behind a
cloudflared tunnel, and expose the identical HTTP contract:

* ``POST /generate`` - text-to-video  (returns MP4 bytes)
* ``POST /image``    - image-to-video (only on >=30 GB GPUs; else HTTP 501)

Paste the PUBLIC ``*.trycloudflare.com`` URL printed by the notebook's tunnel
cell into ``.env`` as ``KAGGLE_BASE_URL``. Kaggle sessions are ephemeral:
whenever the session restarts, re-run the tunnel cell and update the URL.
"""

from __future__ import annotations

import os

from .colab import ColabBackend


class KaggleBackend(ColabBackend):
    """Wan T2V (+ I2V on large GPUs) served from a Kaggle session via public URL."""

    name = "kaggle"
    _out_dir = "outputs/kaggle"
    _hint = (
        "kaggle backend needs KAGGLE_BASE_URL: run the Kaggle server notebook "
        "on a Kaggle GPU session and copy the printed trycloudflare URL into .env"
    )

    def __init__(self, **env):
        super().__init__(**env)
        # Identical API contract as Colab, so a URL pasted as COLAB_BASE_URL
        # from the Kaggle notebook keeps working when KAGGLE_BASE_URL is unset.
        self.base_url = (
            (os.getenv("KAGGLE_BASE_URL", "") or os.getenv("COLAB_BASE_URL", ""))
            or ""
        ).rstrip("/")
        try:
            self.timeout = int(
                os.getenv("KAGGLE_TIMEOUT", "") or os.getenv("COLAB_TIMEOUT", "") or "3600"
            )
        except ValueError:
            self.timeout = 3600
