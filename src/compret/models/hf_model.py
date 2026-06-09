"""Hugging Face transformers backend — CLIP / SigLIP / SigLIP 2 / Jina-CLIP, etc.

Spec: ``hf:<model_id>``. Examples:
    hf:openai/clip-vit-large-patch14-336
    hf:google/siglip2-so400m-patch16-384
    hf:google/siglip-base-patch16-224
    hf:jinaai/jina-clip-v2

Dispatches to ``get_text_features``/``get_image_features`` (CLIP/SigLIP family) or to a
model's own ``encode_text``/``encode_image`` (e.g. Jina, via trust_remote_code).

Requires: ``uv sync --extra hf``.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .base import RetrievalModel, l2norm, register
from ._torchutil import batched, pick_device

_BATCH = 16


class HFModel:
    def __init__(self, model_id: str) -> None:
        if not model_id:
            raise ValueError("hf spec needs a model id, e.g. hf:google/siglip2-base-patch16-224")
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.device = pick_device(torch)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
        self._native = hasattr(self.model, "encode_text") and hasattr(self.model, "encode_image")
        self.processor = None if self._native else AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.name = f"hf:{model_id}"
        self.dim = int(self.encode_text(["x"]).shape[-1])

    def encode_text(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for chunk in batched(list(texts), _BATCH):
                if self._native:
                    feats = self.model.encode_text(list(chunk))
                    out.append(np.asarray(_to_numpy(feats, torch)))
                else:
                    inp = self.processor(
                        text=list(chunk), padding="max_length", truncation=True, return_tensors="pt"
                    ).to(self.device)
                    feats = self.model.get_text_features(**inp)
                    out.append(feats.float().cpu().numpy())
        return l2norm(np.concatenate(out, axis=0))

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for chunk in batched(list(images), _BATCH):
                rgb = [im.convert("RGB") for im in chunk]
                if self._native:
                    feats = self.model.encode_image(rgb)
                    out.append(np.asarray(_to_numpy(feats, torch)))
                else:
                    inp = self.processor(images=rgb, return_tensors="pt").to(self.device)
                    feats = self.model.get_image_features(**inp)
                    out.append(feats.float().cpu().numpy())
        return l2norm(np.concatenate(out, axis=0))


def _to_numpy(feats, torch):
    if isinstance(feats, torch.Tensor):
        return feats.float().cpu().numpy()
    return np.asarray(feats, dtype=np.float32)


@register("hf", "transformers CLIP/SigLIP/SigLIP2/Jina-CLIP via model id. extra: hf")
def _build(rest: str) -> RetrievalModel:
    return HFModel(rest)
