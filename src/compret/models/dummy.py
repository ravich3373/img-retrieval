"""Dependency-free baseline: deterministic hashed embeddings.

Text and images are hashed into *separate* random subspaces, so cross-modal similarity is
essentially chance. It is NOT a meaningful retriever — its job is to validate the whole
pipeline (generate -> encode -> score -> metrics -> results) with only numpy + pillow
installed, before you `uv sync --extra openclip`/`--extra hf`.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
from PIL import Image

from .base import RetrievalModel, l2norm, register

_DIM = 256
_TOK = re.compile(r"[a-z]+")


def _hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


class DummyModel:
    def __init__(self) -> None:
        self.name = "dummy"
        self.dim = _DIM

    def encode_text(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), _DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _TOK.findall(t.lower()):
                h = _hash("txt:" + tok)
                out[i, h % _DIM] += 1.0 if (h >> 8) & 1 else -1.0
        return l2norm(out)

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        out = np.zeros((len(images), _DIM), dtype=np.float32)
        for i, im in enumerate(images):
            small = np.asarray(im.convert("RGB").resize((12, 12)), dtype=np.int32) // 32
            for (y, x, c), v in np.ndenumerate(small):
                h = _hash(f"img:{y},{x},{c},{v}")
                out[i, h % _DIM] += 1.0 if (h >> 8) & 1 else -1.0
        return l2norm(out)


@register("dummy", "Hashed plumbing baseline (no deps; chance-level retrieval).")
def _build(_: str) -> RetrievalModel:
    return DummyModel()
