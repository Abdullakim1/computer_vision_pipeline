"""Backend registry + factory."""

from __future__ import annotations

from ..types import GenerationRequest  # re-export


class BackendNotFound(LookupError):
    """Raised when an unknown backend is requested."""


class BackendRegistry:
    """A small registry of available generator backends."""

    def __init__(self):
        self._backends = {}

    def register(self, cls):
        self._backends[cls.name] = cls
        return cls

    def names(self):
        return tuple(sorted(self._backends))

    def get(self, name):
        try:
            return self._backends[name]
        except KeyError as exc:  # pragma: no cover
            raise BackendNotFound(f"backend not registered: {name!r}") from exc


registry = BackendRegistry()


def create_backend(name: str, **kw):
    """Instantiate a backend by name, importing the module lazily on first use."""
    sources = {
        "cinematic": ("cinematic", "CinematicBackend"),
        "local": ("local", "LocalLatentBackend"),
        "colab": ("colab", "ColabBackend"),
        "luma": ("luma", "LumaBackend"),
        "seedance": ("seedance", "SeedanceBackend"),
        "kling": ("kling", "KlingBackend"),
        "veo": ("veo", "VeoBackend"),
    }
    if name not in sources:
        raise BackendNotFound(
            f"unknown backend {name!r} (have {registry.names()})"
        )
    mod, cls_name = sources[name]
    module = __import__(__name__, fromlist=[mod])
    module = getattr(module, mod)  # import submodule lazily
    Cls = getattr(module, cls_name)
    registry.register(Cls)
    return Cls(**kw)
