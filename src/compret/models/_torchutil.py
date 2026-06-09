"""Shared torch helpers for the local backends (lazy — torch only imported by callers)."""
from __future__ import annotations


def pick_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


def batched(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
