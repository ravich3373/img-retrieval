"""RetrievalModel protocol + a small spec-string registry."""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np
from PIL import Image


@runtime_checkable
class RetrievalModel(Protocol):
    """A dual-encoder that maps text and images into one shared, comparable space."""

    name: str  # canonical spec, e.g. "hf:google/siglip2-base-patch16-224"
    dim: int

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """-> float32 array [len(texts), dim], L2-normalized."""
        ...

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        """-> float32 array [len(images), dim], L2-normalized."""
        ...


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


# prefix -> factory(spec_remainder) -> RetrievalModel
_FACTORIES: dict[str, Callable[[str], RetrievalModel]] = {}
_DESCRIPTIONS: dict[str, str] = {}


def register(prefix: str, description: str = "") -> Callable[[Callable], Callable]:
    def deco(fn: Callable[[str], RetrievalModel]) -> Callable:
        _FACTORIES[prefix] = fn
        _DESCRIPTIONS[prefix] = description
        return fn

    return deco


def build_model(spec: str) -> RetrievalModel:
    """Instantiate a model from a `prefix:rest` spec (or bare `prefix`)."""
    prefix, _, rest = spec.partition(":")
    if prefix not in _FACTORIES:
        raise ValueError(
            f"unknown model prefix {prefix!r}. Registered: {sorted(_FACTORIES)}"
        )
    model = _FACTORIES[prefix](rest)
    # ensure the canonical spec is recorded as the model name
    if not getattr(model, "name", None):
        model.name = spec  # type: ignore[attr-defined]
    return model


def list_specs() -> dict[str, str]:
    """Registered prefixes -> human description."""
    return dict(_DESCRIPTIONS)
