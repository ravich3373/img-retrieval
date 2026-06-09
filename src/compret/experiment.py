"""Run one experiment, persist a self-describing result, and aggregate runs.

Minimal file-based tracking (per the project's "minimal for now" choice): every run writes
``results/runs/<timestamp>__<model>.json`` containing the full config + environment + report,
so a run is reproducible from its own file. ``summarize`` reads them back into a table.
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

from .eval.retrieval import evaluate, load_manifest
from .models import build_model


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _safe(spec: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in spec).strip("-")


def run_experiment(
    model_spec: str,
    data_dir: str | Path,
    results_dir: str | Path = "results",
    ks: tuple[int, ...] = (1, 5, 10),
    notes: str = "",
) -> tuple[dict, Path]:
    data_dir = Path(data_dir)
    manifest = load_manifest(data_dir)

    model = build_model(model_spec)
    report = evaluate(model, data_dir, ks=ks)

    ts = datetime.now()
    record = {
        "timestamp": ts.isoformat(timespec="seconds"),
        "model_spec": model_spec,
        "data_dir": str(data_dir),
        "dataset_meta": manifest.get("meta", {}),
        "ks": list(ks),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "notes": notes,
        "report": report,
    }

    runs_dir = Path(results_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{ts:%Y%m%d-%H%M%S}__{_safe(model_spec)}.json"
    path.write_text(json.dumps(record, indent=2))
    return record, path


def load_runs(results_dir: str | Path = "results") -> list[dict]:
    runs_dir = Path(results_dir) / "runs"
    if not runs_dir.exists():
        return []
    runs = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            runs.append(json.loads(p.read_text()))
        except Exception:
            continue
    return runs


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)


def format_table(runs: list[dict]) -> str:
    """A compact comparison table across runs. Headline = overall retrieval +
    per-swap-type 2AFC accuracy (does the base caption beat the one-factor distractor)."""
    if not runs:
        return "(no runs yet — run `compret run ...`)"
    swap_types = ["binding", "color", "shape", "relation"]
    head = ["model", "data", "R@1", "R@5", "MRR"] + [f"{t}↑" for t in swap_types]
    rows = [head]
    for r in runs:
        rep = r["report"]
        ov = rep["overall"]
        row = [
            r["model_spec"],
            Path(r["data_dir"]).name,
            _fmt(ov.get("R@1")),
            _fmt(ov.get("R@5")),
            _fmt(ov.get("MRR")),
        ]
        mp = rep.get("minimal_pairs", {})
        for t in swap_types:
            acc = mp.get(t, {}).get("t2i_2afc", {}).get("acc")
            row.append(_fmt(acc) if acc is not None else "-")
        rows.append(row)

    widths = [max(len(rows[r][c]) for r in range(len(rows))) for c in range(len(head))]
    lines = []
    for ri, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[c]) for c, cell in enumerate(row)))
        if ri == 0:
            lines.append("  ".join("-" * widths[c] for c in range(len(head))))
    note = (
        "\nLegend: swap columns = 2AFC accuracy (base caption ranks the true image above a "
        "one-factor distractor; 0.5 = chance). 'binding' (swap which object owns the color) "
        "is the key attribute-binding probe. See per-run JSON for Winoground group scores."
    )
    return "\n".join(lines) + "\n" + note
