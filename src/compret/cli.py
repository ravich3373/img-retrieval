"""compret command-line interface.

    compret gen --out data/synth_v1 --n 200 --objects 1,2 --seed 0
    compret models
    compret run --model openclip:ViT-B-32:laion2b_s34b_b79k --data data/synth_v1
    compret summarize
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_gen(args: argparse.Namespace) -> int:
    from .data.generate import generate

    object_counts = tuple(int(x) for x in args.objects.split(","))
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    manifest = generate(
        args.out, n=args.n, object_counts=object_counts, seed=args.seed, variants=variants
    )
    m = manifest["meta"]
    print(f"Wrote {m['n_items']} images to {args.out} "
          f"({args.n} base scenes, objects={object_counts}, variants={variants}, seed={args.seed})")
    return 0


def _cmd_models(_: argparse.Namespace) -> int:
    from .models import list_specs

    print("Model backends (spec = prefix:rest):\n")
    for prefix, desc in sorted(list_specs().items()):
        print(f"  {prefix:10s}  {desc}")
    print("\nExamples:")
    for ex in (
        "dummy",
        "openclip:ViT-L-14-336:openai",
        "openclip:ViT-B-32:laion2b_s34b_b79k",
        "hf:google/siglip2-so400m-patch16-384",
        "hf:jinaai/jina-clip-v2",
        "voyage:voyage-multimodal-3",
    ):
        print(f"  {ex}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .experiment import run_experiment

    ks = tuple(int(x) for x in args.ks.split(","))
    record, path = run_experiment(
        args.model, args.data, results_dir=args.results, ks=ks, notes=args.notes
    )
    rep = record["report"]
    print(f"\n== {args.model}  on  {args.data} ==")
    print(f"items={rep['n_items']} families={rep['n_families']} dim={rep['dim']}")
    print("overall:", json.dumps(rep["overall"]))
    print("by_complexity:")
    for k, v in rep["by_complexity"].items():
        print(f"  {k:10s} {json.dumps(v)}")
    print("minimal pairs (2AFC acc; 0.5=chance):")
    for t, d in rep["minimal_pairs"].items():
        wg = d["winoground"]
        print(f"  {t:9s} t2i={d['t2i_2afc']['acc']:.3f} i2t={d['i2t_2afc']['acc']:.3f} "
              f"| wino text={wg['text_score']:.3f} image={wg['image_score']:.3f} "
              f"group={wg['group_score']:.3f} (n={wg['n']})")
    print(f"\nsaved -> {path}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    import yaml

    from .experiment import format_table, load_runs, run_experiment

    cfg = yaml.safe_load(Path(args.config).read_text())
    data = cfg["data"]
    results = cfg.get("results", "results")
    ks = tuple(cfg.get("ks", [1, 5, 10]))
    models = cfg["models"]
    print(f"benchmarking {len(models)} model(s) on {data}\n")
    for spec in models:
        try:
            print(f"--- {spec} ---")
            run_experiment(spec, data, results_dir=results, ks=ks, notes=cfg.get("notes", ""))
        except Exception as e:  # one bad model shouldn't kill the batch
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    print("\n" + format_table([r for r in load_runs(results)
                               if Path(r["data_dir"]).name == Path(data).name]))
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    from .experiment import format_table, load_runs

    runs = load_runs(args.results)
    if args.model:
        runs = [r for r in runs if args.model in r["model_spec"]]
    if args.data:
        runs = [r for r in runs if Path(r["data_dir"]).name == args.data]
    print(format_table(runs))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="compret", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate a synthetic dataset")
    g.add_argument("--out", required=True, help="output dataset dir")
    g.add_argument("--n", type=int, default=200, help="number of base scenes")
    g.add_argument("--objects", default="1,2", help="comma list of object counts to sample from")
    g.add_argument("--variants", default="binding,color,shape,relation", help="minimal-pair swap types")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=_cmd_gen)

    m = sub.add_parser("models", help="list model backends")
    m.set_defaults(func=_cmd_models)

    r = sub.add_parser("run", help="run a model on a dataset and save results")
    r.add_argument("--model", required=True, help="model spec, e.g. hf:google/siglip2-base-patch16-224")
    r.add_argument("--data", required=True, help="dataset dir (containing manifest.json)")
    r.add_argument("--results", default="results", help="results dir")
    r.add_argument("--ks", default="1,5,10", help="comma list of K for Recall@K")
    r.add_argument("--notes", default="", help="free-text note stored in the run record")
    r.set_defaults(func=_cmd_run)

    b = sub.add_parser("bench", help="run many models on one dataset from a YAML config")
    b.add_argument("--config", required=True, help="YAML with keys: data, models, [results, ks, notes]")
    b.set_defaults(func=_cmd_bench)

    s = sub.add_parser("summarize", help="compare saved runs in a table")
    s.add_argument("--results", default="results")
    s.add_argument("--model", default="", help="filter: substring of model spec")
    s.add_argument("--data", default="", help="filter: dataset dir name")
    s.set_defaults(func=_cmd_summarize)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
