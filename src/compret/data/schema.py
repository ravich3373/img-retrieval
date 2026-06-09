"""Structured scene representation + controlled attribute vocabulary.

A Scene is a small set of Objects placed left->right on a canvas. Every object carries
qualifying adjectives (color, size) and an identity (shape). Because we *render* from this
structure, the ground truth is exact, which is what lets us build clean hard negatives
(swap a color, a shape, the binding, or the relation) for the compositionality probe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --- Controlled vocabulary -------------------------------------------------------------
# Keep palettes well-separated in RGB so "red" vs "blue" is unambiguous in pixel space;
# the point of the study is language/binding, not perceptual color confusion.
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (214, 39, 40),
    "green": (44, 160, 44),
    "blue": (31, 119, 180),
    "yellow": (240, 200, 20),
    "purple": (148, 103, 189),
    "orange": (255, 127, 14),
    "cyan": (23, 190, 207),
    "pink": (227, 119, 194),
}

SHAPES: tuple[str, ...] = ("circle", "square", "triangle", "star", "diamond")

# rendered radius in pixels per size adjective
SIZES: dict[str, int] = {"small": 26, "large": 52}

# relations are derived from left->right placement order (index 0 is leftmost)
RELATIONS: tuple[str, ...] = ("to the left of", "to the right of")


@dataclass(frozen=True)
class Obj:
    """One object with its qualifying adjectives and identity."""

    shape: str
    color: str
    size: str

    def phrase(self) -> str:
        """Noun phrase, e.g. 'a small red circle'. Adjectives = size + color."""
        article = "an" if self.size[0] in "aeiou" else "a"
        return f"{article} {self.size} {self.color} {self.shape}"

    def as_dict(self) -> dict:
        return {"shape": self.shape, "color": self.color, "size": self.size}


@dataclass
class Scene:
    """An ordered (left->right) list of objects + canvas size. Order encodes the relation."""

    objects: list[Obj]
    width: int = 384
    height: int = 256
    scene_id: str = ""
    # ids of rendered minimal-pair partners, keyed by swap type (e.g. {"binding": "0007"})
    partners: dict[str, str] = field(default_factory=dict)

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    @property
    def complexity(self) -> str:
        """Bucket used for per-complexity reporting (CREPE-style)."""
        if self.n_objects == 1:
            return "1obj"
        if self.n_objects == 2:
            return "2obj+rel"
        return f"{self.n_objects}obj+rel"

    def caption(self) -> str:
        """Render the scene as a composed text query.

        1 object:  'a small red circle'
        2 objects: 'a small red circle to the left of a large blue square'
        n objects: chained with ', and ... to the left of ...'
        """
        if not self.objects:
            return ""
        if len(self.objects) == 1:
            return self.objects[0].phrase()
        parts = [self.objects[0].phrase()]
        for obj in self.objects[1:]:
            parts.append(f"to the left of {obj.phrase()}")
        return " ".join(parts)

    def as_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "width": self.width,
            "height": self.height,
            "objects": [o.as_dict() for o in self.objects],
            "complexity": self.complexity,
            "partners": dict(self.partners),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        scene = cls(
            objects=[Obj(**o) for o in d["objects"]],
            width=d.get("width", 384),
            height=d.get("height", 256),
            scene_id=d.get("scene_id", ""),
        )
        scene.partners = dict(d.get("partners", {}))
        return scene
