"""``detkit`` command line: init, import-photos, tile, app, split, train, eval, predict, weight, doctor."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import coco as C
from .project import Project, load_env, parse_classes


def _project(a) -> Project:
    return Project.load(a.workdir)


def cmd_init(a) -> int:
    wd = Path(a.workdir)
    if (wd / "project.json").exists() and not a.force:
        print(f"{wd}/project.json exists (use --force to overwrite)")
        return 1
    from .project import FLOWS

    preset = dict(FLOWS[a.flow])
    p = Project(workdir=wd, classes=parse_classes(a.classes), tile=a.tile, overlap=a.overlap,
                crops=a.crops, flow=a.flow, proposer_ckpt=a.proposer_ckpt, **preset)
    p.save()
    print(f"initialised {p.path}: classes={p.classes}")
    return 0


def cmd_tile(a) -> int:
    from .tiling import make_tiles

    p = _project(a)
    crops = json.loads(Path(p.crops).read_text(encoding="utf-8")) if p.crops else None
    man = make_tiles(a.images or p.photos_dir, p.tiles_dir, p.manifest, p.tile, p.overlap, crops)
    print(f"{len(man['sources'])} photos -> {len(man['tiles'])} tiles in {p.tiles_dir}")
    return 0


def _tiles_annotations(p: Project) -> Path:
    """annotations/tiles_coco.json, written by `detkit split --from-reviewed`."""
    if not p.tiles_coco.exists():
        raise SystemExit(f"{p.tiles_coco} not found - use `detkit split --from-reviewed` (needs reviewed photos from the app)")
    return p.tiles_coco


def cmd_split(a) -> int:
    from .split import build_dataset

    p = _project(a)
    if a.from_reviewed:
        from .cmd_data import retile

        cmd_tile(argparse.Namespace(workdir=a.workdir, images=None))   # always re-tile: picks up new photos
        st = retile(p, include_unreviewed=a.include_unreviewed, min_visible=a.min_visible,
                    min_px=a.min_px)
        print(f"retile: {st['photos']} photos, {st['tiles']} tiles, per_class={st['per_class']}")
        if not st["photos"]:
            print("no reviewed photos (review in the app or pass --include-unreviewed)", file=sys.stderr)
            return 1
    val = [v.strip() for v in a.val_sources.split(",") if v.strip()] if a.val_sources else None
    rep = build_dataset(C.load(_tiles_annotations(p)), C.load(p.manifest), p.tiles_dir, p.coco_dir,
                        val_sources=val, val_frac=a.val_frac, seed=a.seed)
    print(json.dumps({k: rep[k] for k in ("val_sources", "totals", "warnings")}, indent=2))
    for w in rep["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0


def cmd_train(a) -> int:
    p = _project(a)
    device = a.device or p.device
    if device == "cpu":                       # hide the GPU BEFORE torch is imported (train is imported lazily)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.push_to_hub and not a.hub_model_id:
        raise SystemExit("--push-to-hub needs --hub-model-id user/repo (and HF_TOKEN with write access)")
    from .train import train_model

    tp = C.load(p.manifest)["params"] if p.manifest.exists() else {"tile": p.tile, "overlap": p.overlap}
    tp = {"tile": tp.get("tile"), "overlap": tp.get("overlap")}
    out = train_model(p.coco_dir, a.out or p.ckpt_dir / a.name, model=a.model or p.train_model,
                      epochs=a.epochs, batch_size=a.batch_size or p.train_batch, lr=a.lr,
                      grad_accum=a.grad_accum, augment=not a.no_augment,
                      num_workers=p.train_workers if a.num_workers is None else a.num_workers,
                      logging_steps=a.logging_steps, load_best=not a.no_load_best, tile_params=tp,
                      push_to_hub=a.push_to_hub, hub_model_id=a.hub_model_id, seed=a.seed,
                      eval_batch_size=a.eval_batch_size, weight_decay=a.weight_decay)
    print(f"saved {out}")
    return 0


def cmd_predict(a) -> int:
    from .predict import predict_paths

    crop = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
    predict_paths(a.model, a.images, a.out, tile=a.tile, overlap=a.overlap, crop=crop,
                  score=a.score, batch=a.batch, threads=a.threads, max_side=a.max_side,
                  overlay=not a.no_overlay)
    return 0


def cmd_doctor(a) -> int:
    from .doctor import check_gpu, check_token

    print(json.dumps({"token": check_token(a.online), "gpu": check_gpu()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="detkit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, workdir=True):
        sp = sub.add_parser(name, help=help_)
        if workdir:
            sp.add_argument("--workdir", default="runs/default")
        sp.set_defaults(fn=fn)
        return sp

    s = add("init", cmd_init, "create <workdir>/project.json")
    s.add_argument("--classes", required=True, help="comma-separated, e.g. corn,wheat")
    s.add_argument("--tile", type=int, default=800)
    s.add_argument("--overlap", type=float, default=0.2)
    s.add_argument("--crops", help="crops.json (see detkit/tiling.py)")
    s.add_argument("--flow", choices=["hf", "cpu", "gpu"], default="hf",
                   help="where each step runs (see docs/TRAINING.md)")
    s.add_argument("--proposer-ckpt", help="RF-DETR checkpoint dir used for box proposals")
    s.add_argument("--force", action="store_true")

    s = add("tile", cmd_tile, "cut photos into overlapping tiles + manifest")
    s.add_argument("--images", help="directory of source photos (default: <workdir>/photos)")

    s = add("split", cmd_split, "build <workdir>/coco train/val split by source photo")
    s.add_argument("--val-sources", help="comma-separated source photo file names for val")
    s.add_argument("--val-frac", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--from-reviewed", action="store_true",
                   help="first (re)tile W/photos and cut annotations/reviewed_coco.json into tiles_coco.json")
    s.add_argument("--include-unreviewed", action="store_true", help="with --from-reviewed")
    s.add_argument("--min-visible", type=float, default=0.6, help="with --from-reviewed")
    s.add_argument("--min-px", type=float, default=8.0, help="with --from-reviewed")

    s = add("train", cmd_train, "train RF-DETR on <workdir>/coco")
    s.add_argument("--model", help="nano|small|base|medium|large or HF id (default: flow preset)")
    s.add_argument("--name", default="run1")
    s.add_argument("--out")
    s.add_argument("--epochs", type=int, default=30)
    s.add_argument("--batch-size", type=int, help="default: flow preset")
    s.add_argument("--grad-accum", type=int, default=1)
    s.add_argument("--lr", type=float, default=5e-5)
    s.add_argument("--no-augment", action="store_true")
    s.add_argument("--no-load-best", action="store_true",
                   help="final/ = last epoch instead of the best-eval checkpoint")
    s.add_argument("--num-workers", type=int, help="default: flow preset")
    s.add_argument("--logging-steps", type=int, default=5)
    s.add_argument("--seed", type=int, default=1337)
    s.add_argument("--eval-batch-size", type=int, help="default: --batch-size")
    s.add_argument("--weight-decay", type=float, default=1e-4)
    s.add_argument("--device", choices=["auto", "cpu", "cuda"],
                   help="cpu hides the GPU (default: project.json 'device', set by `init --flow`)")
    s.add_argument("--push-to-hub", action="store_true",
                   help="HF: upload the trained model to the Hub (needs --hub-model-id and a write token)")
    s.add_argument("--hub-model-id", help="e.g. my-user/grain-detector")

    s = add("predict", cmd_predict, "tiled inference on full photos", workdir=False)
    s.add_argument("--model", required=True, help="checkpoint dir (…/final)")
    s.add_argument("--images", nargs="+", required=True)
    s.add_argument("--out", default="predictions")
    s.add_argument("--score", type=float, default=0.5)
    s.add_argument("--tile", type=int)
    s.add_argument("--overlap", type=float)
    s.add_argument("--crop", help="x0,y0,x1,y1 region to scan (default: whole photo)")
    s.add_argument("--batch", type=int, default=4, help="tiles per forward pass")
    s.add_argument("--threads", type=int, help="torch CPU threads")
    s.add_argument("--max-side", type=int,
                   help="downscale the photo to this longest side before tiling (fast on CPU; "
                        "tile/crop then refer to the downscaled image)")
    s.add_argument("--no-overlay", action="store_true", help="skip drawing overlay images")

    s = add("doctor", cmd_doctor, "check GPU/torch and HF token", workdir=False)
    s.add_argument("--online", action="store_true", help="also ping HF (no secrets printed)")
    _load_plugins(sub)
    return ap


def _load_plugins(sub) -> None:
    """Every ``detkit/cmd_<name>.py`` may define ``register(sub)`` to add its own subcommands."""
    import importlib
    import pkgutil

    import detkit

    for m in sorted(pkgutil.iter_modules(detkit.__path__), key=lambda m: m.name):
        if m.name.startswith("cmd_"):
            mod = importlib.import_module(f"detkit.{m.name}")
            if hasattr(mod, "register"):
                mod.register(sub)


def main(argv: list[str] | None = None) -> int:
    load_env()
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
