"""Pluggable retrieval-model backends, addressed by a `prefix:rest` spec string.

Examples:
    dummy
    openclip:ViT-B-32:laion2b_s34b_b79k
    hf:google/siglip2-base-patch16-224
    hf:openai/clip-vit-large-patch14-336
    voyage:voyage-multimodal-3
    cohere:embed-v4.0

Backends import their heavy deps lazily, so `import compret.models` is cheap and the
`dummy` backend always works even with nothing but numpy + pillow installed.
"""
from .base import RetrievalModel, build_model, list_specs, register, l2norm

# Import side-effect: register built-in backends.
from . import dummy  # noqa: F401
from . import openclip_model  # noqa: F401
from . import hf_model  # noqa: F401

# Hosted API-only backends are parked for now (we're testing open-weight models first).
# Re-enable by uncommenting; the code in hosted.py still works (needs keys + verification).
# from . import hosted  # noqa: F401, ERA001

__all__ = ["RetrievalModel", "build_model", "list_specs", "register", "l2norm"]
