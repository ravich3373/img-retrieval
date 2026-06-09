"""Render a controlled synthetic dataset and write a manifest with exact ground truth.

For each base scene we also render single-factor *minimal-pair* variants so the gallery
contains genuine hard negatives:

    binding  - swap the two objects' colors (same words, different binding)   [2+ objs]
    color    - change one object's color to an absent color                    [attribute value]
    shape    - change one object's shape                                       [object identity]
    relation - reverse left/right order                                        [2+ objs]

Every caption is unique across the whole gallery, so each query has exactly one gold image.
Determinism is seeded for reproducible experiment tracking.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

from .schema import COLORS, RELATIONS, SHAPES, SIZES, Obj, Scene

_SS = 2  # supersampling factor for anti-aliased edges
_OUTLINE = (40, 40, 40)
_BG = (255, 255, 255)


# --- rendering -------------------------------------------------------------------------
def _star_points(cx: float, cy: float, r: float, inner_ratio: float = 0.45) -> list[tuple[float, float]]:
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * inner_ratio
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    return pts


def _draw_obj(draw: ImageDraw.ImageDraw, obj: Obj, cx: float, cy: float) -> None:
    r = SIZES[obj.size] * _SS
    fill = COLORS[obj.color]
    ow = max(2, _SS)
    if obj.shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=_OUTLINE, width=ow)
    elif obj.shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=_OUTLINE, width=ow)
    elif obj.shape == "triangle":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill, outline=_OUTLINE, width=ow)
    elif obj.shape == "diamond":
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=fill, outline=_OUTLINE, width=ow)
    elif obj.shape == "star":
        draw.polygon(_star_points(cx, cy, r), fill=fill, outline=_OUTLINE, width=ow)
    else:  # pragma: no cover - vocab is closed
        raise ValueError(f"unknown shape {obj.shape!r}")


def render_scene(scene: Scene, rng: random.Random | None = None) -> Image.Image:
    """Render a scene to a PIL image. Objects are laid out left->right in equal cells."""
    W, H = scene.width * _SS, scene.height * _SS
    img = Image.new("RGB", (W, H), _BG)
    draw = ImageDraw.Draw(img)
    n = scene.n_objects
    for i, obj in enumerate(scene.objects):
        cell_w = W / n
        cx = cell_w * (i + 0.5)
        cy = H / 2
        # small deterministic jitter so positions aren't pixel-identical across scenes
        if rng is not None:
            cx += rng.uniform(-cell_w * 0.08, cell_w * 0.08)
            cy += rng.uniform(-H * 0.06, H * 0.06)
        _draw_obj(draw, obj, cx, cy)
    return img.resize((scene.width, scene.height), Image.LANCZOS)


# --- single-factor variants ------------------------------------------------------------
def _swap_binding(s: Scene) -> Scene | None:
    if s.n_objects < 2 or s.objects[0].color == s.objects[1].color:
        return None
    objs = list(s.objects)
    a, b = objs[0], objs[1]
    objs[0] = Obj(a.shape, b.color, a.size)
    objs[1] = Obj(b.shape, a.color, b.size)
    return Scene(objs, s.width, s.height)


def _change_color(s: Scene, rng: random.Random) -> Scene | None:
    present = {o.color for o in s.objects}
    choices = [c for c in COLORS if c not in present]
    if not choices:
        return None
    objs = list(s.objects)
    objs[0] = Obj(objs[0].shape, rng.choice(choices), objs[0].size)
    return Scene(objs, s.width, s.height)


def _change_shape(s: Scene, rng: random.Random) -> Scene | None:
    choices = [sh for sh in SHAPES if sh != s.objects[0].shape]
    objs = list(s.objects)
    objs[0] = Obj(rng.choice(choices), objs[0].color, objs[0].size)
    return Scene(objs, s.width, s.height)


def _reverse_relation(s: Scene) -> Scene | None:
    if s.n_objects < 2:
        return None
    return Scene(list(reversed(s.objects)), s.width, s.height)


_VARIANTS = {
    "binding": lambda s, rng: _swap_binding(s),
    "color": lambda s, rng: _change_color(s, rng),
    "shape": lambda s, rng: _change_shape(s, rng),
    "relation": lambda s, rng: _reverse_relation(s),
}


# --- scene sampling --------------------------------------------------------------------
def _sample_scene(rng: random.Random, n_objects: int) -> Scene:
    objs = []
    for _ in range(n_objects):
        objs.append(
            Obj(
                shape=rng.choice(SHAPES),
                color=rng.choice(list(COLORS)),
                size=rng.choice(list(SIZES)),
            )
        )
    width = 384 if n_objects <= 2 else 192 * n_objects
    return Scene(objs, width=width, height=256)


def generate(
    out_dir: str | Path,
    n: int = 200,
    object_counts: tuple[int, ...] = (1, 2),
    seed: int = 0,
    variants: tuple[str, ...] = ("binding", "color", "shape", "relation"),
) -> dict:
    """Generate the dataset under out_dir. Returns the manifest dict (also written to disk)."""
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    items: list[dict] = []
    families: list[dict] = []
    seen_captions: set[str] = set()
    counter = 0

    def _add(scene: Scene, role: str, parent: str | None, vtype: str | None) -> dict | None:
        nonlocal counter
        cap = scene.caption()
        if cap in seen_captions:
            return None  # keep captions globally unique -> one gold per query
        seen_captions.add(cap)
        sid = f"{counter:05d}"
        counter += 1
        scene.scene_id = sid
        render_rng = random.Random(f"{seed}-{counter}")
        render_scene(scene, render_rng).save(img_dir / f"{sid}.png")
        item = {
            "id": sid,
            "image": f"images/{sid}.png",
            "caption": cap,
            "scene": scene.as_dict(),
            "role": role,
            "parent_id": parent,
            "variant_type": vtype,
        }
        items.append(item)
        return item

    for _ in range(n):
        nobj = rng.choice(object_counts)
        base = _sample_scene(rng, nobj)
        base_item = _add(base, role="base", parent=None, vtype=None)
        if base_item is None:
            continue
        base_id = base_item["id"]
        partner_ids: dict[str, str] = {}
        for vtype in variants:
            var_scene = _VARIANTS[vtype](base, rng)
            if var_scene is None:
                continue
            var_item = _add(var_scene, role="variant", parent=base_id, vtype=vtype)
            if var_item is not None:
                partner_ids[vtype] = var_item["id"]
        base_item["scene"]["partners"] = partner_ids
        families.append({"base_id": base_id, "variants": partner_ids})

    manifest = {
        "meta": {
            "n_base_requested": n,
            "n_items": len(items),
            "object_counts": list(object_counts),
            "variants": list(variants),
            "seed": seed,
            "colors": list(COLORS),
            "shapes": list(SHAPES),
            "sizes": list(SIZES),
            "relations": list(RELATIONS),
        },
        "items": items,
        "families": families,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
