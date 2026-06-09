"""Pure-numpy retrieval metrics.

Convention everywhere: ``sim`` is a [Q, N] matrix where row q is a text query and column n
is a gallery image; higher = more similar (embeddings are L2-normalized, so sim is cosine).
``gold`` is an int array [Q] giving the gold image column for each query.
"""
from __future__ import annotations

import numpy as np


def ranks_of_gold(sim: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """1-based rank of the gold image for each query (1 = retrieved first)."""
    gold_score = sim[np.arange(sim.shape[0]), gold][:, None]
    # rank = 1 + (# candidates strictly better than gold)
    return 1 + (sim > gold_score).sum(axis=1)


def recall_at_k(sim: np.ndarray, gold: np.ndarray, ks=(1, 5, 10)) -> dict[str, float]:
    r = ranks_of_gold(sim, gold)
    return {f"R@{k}": float((r <= k).mean()) for k in ks}


def mrr(sim: np.ndarray, gold: np.ndarray) -> float:
    return float((1.0 / ranks_of_gold(sim, gold)).mean())


def median_rank(sim: np.ndarray, gold: np.ndarray) -> float:
    return float(np.median(ranks_of_gold(sim, gold)))


def two_way_accuracy(pos: np.ndarray, neg: np.ndarray) -> dict[str, float]:
    """Accuracy of preferring pos over neg, given paired score arrays."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0:
        return {"acc": float("nan"), "n": 0}
    return {"acc": float((pos > neg).mean()), "n": int(len(pos))}


def winoground_scores(s_bb, s_bp, s_pb, s_pp) -> dict[str, float]:
    """Winoground text/image/group scores for minimal pairs.

    Indices: s_xy = sim(caption x, image y), x,y in {b(ase), p(artner)}. Each input is an
    array over pairs. text_score: each image's correct caption wins. image_score: each
    caption's correct image wins. group: both.
    """
    s_bb, s_bp, s_pb, s_pp = map(np.asarray, (s_bb, s_bp, s_pb, s_pp))
    n = len(s_bb)
    if n == 0:
        return {"text_score": float("nan"), "image_score": float("nan"), "group_score": float("nan"), "n": 0}
    text = (s_bb > s_pb) & (s_pp > s_bp)   # per image, correct caption higher
    image = (s_bb > s_bp) & (s_pp > s_pb)  # per caption, correct image higher
    return {
        "text_score": float(text.mean()),
        "image_score": float(image.mean()),
        "group_score": float((text & image).mean()),
        "n": int(n),
    }
