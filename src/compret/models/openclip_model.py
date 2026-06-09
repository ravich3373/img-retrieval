"""open_clip backend — LAION / OpenAI / DFN / EVA / MetaCLIP / NegCLIP checkpoints.

Spec: ``openclip:<arch>:<pretrained>``  (pretrained defaults to ``openai``).
Examples:
    openclip:ViT-B-32:laion2b_s34b_b79k
    openclip:ViT-L-14-336:openai
    openclip:ViT-H-14-378-quickgelu:dfn5b           # apple DFN5B
    openclip:ViT-bigG-14:laion2b_s39b_b160k
NegCLIP / local checkpoints: ``openclip:ViT-B-32:/abs/path/negclip.pt``.

Requires: ``uv sync --extra openclip``.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .base import RetrievalModel, l2norm, register
from ._torchutil import batched, pick_device

_BATCH = 32


class OpenClipModel:
    def __init__(self, spec: str) -> None:
        import open_clip  # noqa: PLC0415  (lazy heavy import)
        import torch

        arch, _, pretrained = spec.partition(":")
        if not arch:
            raise ValueError("openclip spec needs an arch, e.g. openclip:ViT-B-32:laion2b_s34b_b79k")
        pretrained = pretrained or "openai"
        self._torch = torch
        self.device = pick_device(torch)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.name = f"openclip:{arch}:{pretrained}"
        with torch.no_grad():
            self.dim = int(self.model.encode_text(self.tokenizer(["x"]).to(self.device)).shape[-1])

    def encode_text(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for chunk in batched(texts, _BATCH):
                toks = self.tokenizer(list(chunk)).to(self.device)
                out.append(self.model.encode_text(toks).float().cpu().numpy())
        return l2norm(np.concatenate(out, axis=0))

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for chunk in batched(images, _BATCH):
                px = torch.stack([self.preprocess(im.convert("RGB")) for im in chunk]).to(self.device)
                out.append(self.model.encode_image(px).float().cpu().numpy())
        return l2norm(np.concatenate(out, axis=0))


@register("openclip", "open_clip checkpoints (LAION/OpenAI/DFN/EVA/MetaCLIP/NegCLIP). extra: openclip")
def _build(rest: str) -> RetrievalModel:
    return OpenClipModel(rest)
