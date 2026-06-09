"""Hosted multimodal-embedding API backends: Voyage, Cohere, Jina.

⚠️ UNVERIFIED: this was authored offline (no web access during the build), so the exact
request shapes are best-effort against APIs as known at Jan 2026 and MAY HAVE DRIFTED.
Each provider's call is centralized in one place so you can fix it in one spot. Set the
API key via env var and run a tiny smoke test before trusting numbers.

Specs / env vars:
    voyage:voyage-multimodal-3     VOYAGE_API_KEY      (extra: hosted -> `voyageai`)
    cohere:embed-v4.0              CO_API_KEY          (extra: hosted -> `cohere`)
    jina:jina-clip-v2              JINA_API_KEY        (extra: hosted -> `requests`)
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np
from PIL import Image

from .base import RetrievalModel, l2norm, register

_BATCH = 16


def _png_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# --- Voyage ----------------------------------------------------------------------------
class VoyageModel:
    def __init__(self, model_id: str) -> None:
        import voyageai

        self.model_id = model_id or "voyage-multimodal-3"
        self.client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
        self.name = f"voyage:{self.model_id}"
        self.dim = int(self.encode_text(["x"]).shape[-1])

    def _embed(self, inputs: list[list], input_type: str) -> np.ndarray:
        out = []
        for chunk in _chunks(inputs, _BATCH):
            r = self.client.multimodal_embed(inputs=list(chunk), model=self.model_id, input_type=input_type)
            out.extend(r.embeddings)
        return np.asarray(out, dtype=np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        return l2norm(self._embed([[t] for t in texts], "query"))

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        return l2norm(self._embed([[im.convert("RGB")] for im in images], "document"))


# --- Cohere ----------------------------------------------------------------------------
class CohereModel:
    def __init__(self, model_id: str) -> None:
        import cohere

        self.model_id = model_id or "embed-v4.0"
        self.client = cohere.ClientV2(api_key=os.environ.get("CO_API_KEY") or os.environ.get("COHERE_API_KEY"))
        self.name = f"cohere:{self.model_id}"
        self.dim = int(self.encode_text(["x"]).shape[-1])

    def encode_text(self, texts: list[str]) -> np.ndarray:
        out = []
        for chunk in _chunks(list(texts), _BATCH):
            r = self.client.embed(
                model=self.model_id, input_type="search_query", texts=list(chunk), embedding_types=["float"]
            )
            out.extend(r.embeddings.float)
        return l2norm(np.asarray(out, dtype=np.float32))

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        out = []
        for im in images:  # image endpoint historically took one image per call
            data_uri = f"data:image/png;base64,{_png_b64(im)}"
            r = self.client.embed(
                model=self.model_id, input_type="image", images=[data_uri], embedding_types=["float"]
            )
            out.extend(r.embeddings.float)
        return l2norm(np.asarray(out, dtype=np.float32))


# --- Jina (REST) -----------------------------------------------------------------------
class JinaModel:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id or "jina-clip-v2"
        self.key = os.environ.get("JINA_API_KEY")
        if not self.key:
            raise RuntimeError("set JINA_API_KEY")
        self.name = f"jina:{self.model_id}"
        self.dim = int(self.encode_text(["x"]).shape[-1])

    def _post(self, inputs: list[dict]) -> np.ndarray:
        import requests

        out = []
        for chunk in _chunks(inputs, _BATCH):
            resp = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                json={"model": self.model_id, "input": list(chunk)},
                timeout=120,
            )
            resp.raise_for_status()
            out.extend(d["embedding"] for d in resp.json()["data"])
        return np.asarray(out, dtype=np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        return l2norm(self._post([{"text": t} for t in texts]))

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        return l2norm(self._post([{"image": _png_b64(im)} for im in images]))


@register("voyage", "Voyage multimodal embeddings (UNVERIFIED). env: VOYAGE_API_KEY. extra: hosted")
def _build_voyage(rest: str) -> RetrievalModel:
    return VoyageModel(rest)


@register("cohere", "Cohere Embed multimodal (UNVERIFIED). env: CO_API_KEY. extra: hosted")
def _build_cohere(rest: str) -> RetrievalModel:
    return CohereModel(rest)


@register("jina", "Jina embeddings REST (UNVERIFIED). env: JINA_API_KEY. extra: hosted")
def _build_jina(rest: str) -> RetrievalModel:
    return JinaModel(rest)
