"""Run a model over a dataset and produce the full compositional-retrieval report.

One [N, N] cosine matrix (item caption x item image) yields everything: full-gallery
retrieval, per-complexity / per-role breakdowns, and minimal-pair compositionality scores
(2AFC + Winoground text/image/group) split by swap type.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import metrics as M


def load_manifest(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    return json.loads((data_dir / "manifest.json").read_text())


def _group_metrics(ranks: np.ndarray, idxs: list[int], ks=(1, 5, 10)) -> dict:
    if not idxs:
        return {"n": 0}
    r = ranks[np.asarray(idxs)]
    out = {f"R@{k}": float((r <= k).mean()) for k in ks}
    out["MRR"] = float((1.0 / r).mean())
    out["median_rank"] = float(np.median(r))
    out["n"] = int(len(idxs))
    return out


def evaluate(model, data_dir: str | Path, ks=(1, 5, 10)) -> dict:
    data_dir = Path(data_dir)
    manifest = load_manifest(data_dir)
    items = manifest["items"]
    id2idx = {it["id"]: i for i, it in enumerate(items)}

    captions = [it["caption"] for it in items]
    images = [Image.open(data_dir / it["image"]).convert("RGB") for it in items]

    txt = model.encode_text(captions)          # [N, D]
    img = model.encode_image(images)           # [N, D]
    sim = (txt @ img.T).astype(np.float64)     # [N, N]; row=caption, col=image
    n = len(items)
    gold = np.arange(n)
    ranks = M.ranks_of_gold(sim, gold)

    # full-gallery retrieval + breakdowns (queries restricted, gallery always full)
    by_complexity: dict[str, list[int]] = defaultdict(list)
    by_role: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_complexity[it["scene"]["complexity"]].append(i)
        by_role[it["role"]].append(i)

    # minimal-pair compositionality, split by swap type
    pairs_by_type: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for fam in manifest["families"]:
        b = id2idx[fam["base_id"]]
        for vtype, vid in fam["variants"].items():
            if vid in id2idx:
                pairs_by_type[vtype].append((b, id2idx[vid]))

    minimal_pairs = {}
    for vtype, pairs in sorted(pairs_by_type.items()):
        bi = np.array([p[0] for p in pairs])
        pi = np.array([p[1] for p in pairs])
        s_bb, s_bp = sim[bi, bi], sim[bi, pi]
        s_pb, s_pp = sim[pi, bi], sim[pi, pi]
        minimal_pairs[vtype] = {
            "t2i_2afc": M.two_way_accuracy(s_bb, s_bp),   # base caption: base img > distractor img
            "i2t_2afc": M.two_way_accuracy(s_bb, s_pb),   # base img: base caption > distractor caption
            "winoground": M.winoground_scores(s_bb, s_bp, s_pb, s_pp),
        }

    return {
        "model": getattr(model, "name", "?"),
        "dim": int(getattr(model, "dim", txt.shape[-1])),
        "n_items": n,
        "n_families": len(manifest["families"]),
        "overall": _group_metrics(ranks, list(range(n)), ks),
        "by_complexity": {k: _group_metrics(ranks, v, ks) for k, v in sorted(by_complexity.items())},
        "by_role": {k: _group_metrics(ranks, v, ks) for k, v in sorted(by_role.items())},
        "minimal_pairs": minimal_pairs,
    }
