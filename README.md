# compret — compositional text→image retrieval experiments

**Question under test:** when objects carry *qualifying adjectives* ("a **red** car") and several
objects are *composed* in one query ("a red car **to the left of** a wooden chair"), which
models/techniques actually retrieve the right image — and where does attribute binding break?

This repo is an experiment harness, not a single experiment. It lets you:

1. **Generate** a controlled synthetic image dataset with exact ground truth, plus single-factor
   *minimal-pair* hard negatives (swap which object owns a color, change a color, change a shape,
   reverse a spatial relation).
2. **Run** any of a growing set of retrieval models (open CLIP-family, hosted multimodal APIs)
   through a pluggable interface.
3. **Measure** standard retrieval (Recall@K, MRR) *and* compositionality (per-swap-type 2AFC
   accuracy + Winoground text/image/group scores), broken down by composition complexity.
4. **Track** each run as a self-describing JSON file and compare them in a table.

See **[SURVEY.md](SURVEY.md)** for the model landscape (what people use to encode image + text
query) and the full evaluation protocol this harness implements.

> Heads-up: SURVEY.md was compiled offline (no web access during the build), so HF IDs / pricing /
> benchmark numbers are **unverified** — reconfirm before relying on specifics. The hosted-API
> backends are likewise best-effort and need an API key + a smoke test.

## Setup

System Python here is 3.14 (too new for most ML wheels), so the project pins to 3.12 via `uv`:

```bash
uv venv --python 3.12
uv sync                      # core only: numpy, pillow, pyyaml  (dummy backend works now)
uv sync --extra openclip     # + torch + open_clip_torch + timm
uv sync --extra hf           # + torch + transformers + einops + timm (CLIP/SigLIP/SigLIP 2/Jina/MetaCLIP 2)
uv sync --extra all          # everything
```

> We're testing **open-weight** models for now; the API-only hosted backends (`voyage:`/`cohere:`/
> `jina:`) are parked (`models/hosted.py`, not registered by default — `--extra hosted` to revive).

## Quickstart

```bash
# 1) make a controlled dataset (renders PNGs + manifest.json with ground truth)
uv run compret gen --out data/synth_v1 --n 200 --objects 1,2 --seed 0

# 2) smoke-test the whole pipeline with the dependency-free baseline
uv run compret run --model dummy --data data/synth_v1

# 3) list backends, then run a real model (after `uv sync --extra hf`)
uv run compret models
uv run compret run --model hf:google/siglip2-base-patch16-224 --data data/synth_v1

# 4) sweep many models from a config, then compare
uv run compret bench --config configs/example.yaml
uv run compret summarize
```

## Model specs

A model is addressed by a `prefix:rest` string (run `compret models`):

| spec | backend | extra |
|---|---|---|
| `dummy` | hashed plumbing baseline (chance-level) | none |
| `openclip:<arch>:<pretrained>` | open_clip (LAION/OpenAI/DFN/EVA/NegCLIP); e.g. `ViT-H-14-378-quickgelu:dfn5b` | `openclip` |
| `hf:<model_id>` | transformers (CLIP/SigLIP/SigLIP 2/Jina-CLIP/MetaCLIP 2) | `hf` |
| `voyage:` / `cohere:` / `jina:` | hosted APIs — **deferred**, not registered by default | `hosted` |

Adding a backend = implement `encode_text(list[str])` and `encode_image(list[PIL])` returning
L2-normalized `[N, D]` arrays, then `@register("prefix", ...)`. See `src/compret/models/dummy.py`.

## What gets measured

Per run (full detail in the saved JSON; `compret summarize` shows the headline table):

- **overall** — Recall@{1,5,10}, MRR, median rank over the full gallery.
- **by_complexity** — same metrics split into `1obj` / `2obj+rel` to show degradation with depth.
- **by_role** — base images vs. rendered variant images.
- **minimal_pairs** — per swap type (`binding`, `color`, `shape`, `relation`):
  - `t2i_2afc` / `i2t_2afc` — does the true caption/image beat the one-factor distractor? (0.5 = chance)
  - `winoground` — text / image / **group** score (group is the strictest binding signal).

`binding` (swap which object owns a color, same words) is the key attribute-binding probe.

## Results & tracking

Each run writes `results/runs/<timestamp>__<model>.json` with config, dataset meta, git commit,
platform, notes, and the full report — reproducible from its own file. `results/runs/` is
gitignored (bulky); commit curated summaries if you want them tracked. `compret summarize`
tabulates everything under `results/`.

## Layout

```
src/compret/
  data/        schema.py (vocab + Scene), generate.py (render + minimal pairs + manifest)
  models/      base.py (protocol+registry), dummy / openclip_model / hf_model / hosted
  eval/        metrics.py (recall/mrr/2afc/winoground), retrieval.py (orchestrator)
  experiment.py  run + persist + summarize
  cli.py       gen / models / run / bench / summarize
configs/       benchmark YAMLs
data/          generated datasets (gitignored)
results/       run outputs (runs/ gitignored)
SURVEY.md      model landscape + evaluation protocol
```

## Roadmap / ideas

- More attribute axes (material, count, texture) and >2-object composition.
- Adversarial text-blind filter (SugarCrepe-style) to certify negatives aren't language-guessable.
- Composed image retrieval (image+text query): MagicLens / CIReVL backends.
- Photoreal datasets (diffusion-generated or real surveillance frames) alongside the synthetic set.
- Video corpus + TwelveLabs Marengo (closer to the production use case).
