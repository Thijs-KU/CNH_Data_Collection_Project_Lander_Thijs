"""``detkit weight predict|calibrate``: photos' detections -> weight estimates (docs/WEIGHT.md)."""

from __future__ import annotations

from pathlib import Path

from .project import Project


def _samples(a, use: str | None = None):
    from .weight import SampleTable

    use = use or ("predictions" if getattr(a, "predictions", None) else a.use)
    return SampleTable.from_project(Project.load(a.workdir), use=use,
                                    predictions_path=getattr(a, "predictions", None),
                                    reviewed_only=getattr(a, "reviewed_only", False))


def _f(v, fmt="{:.3f}") -> str:
    return "-" if v is None else fmt.format(v)


def configured_model(cfg: dict, name: str):
    """Model *name* with its constants from weight_config.json (models without a config_key: fresh)."""
    from .weight import MODELS, get_model

    key = getattr(MODELS.get(name), "config_key", None)
    return get_model(name, constants=cfg[key]) if key else get_model(name)


def cmd_predict(a) -> int:
    from .weight import evaluate_model, load_config

    cfg = load_config(a.workdir)
    name = a.model or cfg["model"]
    samples = _samples(a)
    try:
        model = configured_model(cfg, name)
    except KeyError as ex:
        print(ex)
        return 1
    try:
        if not getattr(model, "config_key", None):
            model.fit(samples.with_weight())
        recs = []
        for s in samples:
            e = model.predict(s.features, s.crop)
            r = {"file": s.file, "crop": s.crop, "count": s.features["count"], "est_g": round(e.weight_g, 3),
                 "true_g": s.weight_g, "error_g": None, "error_pct": None, "note": e.note}
            if s.weight_g:
                r["error_g"] = round(e.weight_g - s.weight_g, 3)
                r["error_pct"] = round(100 * r["error_g"] / s.weight_g, 1)
            recs.append(r)
    except NotImplementedError as ex:
        print(f"model {name!r} is a student TODO: {ex}")
        return 2
    except ValueError as ex:                                   # e.g. area_x_constant not calibrated yet
        print(ex)
        return 1
    print(f"model={name}  photos={len(recs)}")
    print(f"{'file':34s} {'crop':10s} {'count':>5s} {'est_g':>8s} {'true_g':>8s} {'err_%':>7s}")
    for r in recs:
        print(f"{r['file'][-34:]:34s} {str(r['crop']):10s} {r['count']:5d} {r['est_g']:8.3f} "
              f"{_f(r['true_g']):>8s} {_f(r['error_pct'], '{:.1f}'):>7s}")
    ev = evaluate_model(model, samples)
    if ev["n"]:
        print(f"vs balance (n={ev['n']}): MAE {ev['mae_g']:.3f} g, bias {ev['bias_g']:+.3f} g, "
              f"MAPE {_f(ev['mape'] and 100 * ev['mape'], '{:.1f}')} %")
    if a.out:
        import csv
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0])) if recs else None
            if w:
                w.writeheader()
                w.writerows(recs)
        print(f"wrote {a.out}")
    return 0


def cmd_calibrate(a) -> int:
    from .weight import load_config, save_config

    cfg = load_config(a.workdir)
    samples = _samples(a).with_weight()
    model = configured_model(cfg, a.model).fit(samples)
    cfg.update({"model": a.model, model.config_key: model.constants})
    p = save_config(a.workdir, cfg)
    print(f"calibrated {a.model} on {len(samples)} weighed photos -> {p} (default model is now {a.model})")
    for k, v in model.constants.items():
        print(f"  {k:10s} {v:.6g} {model.unit}")
    return 0


def register(sub) -> None:
    sp = sub.add_parser("weight", help="estimate weight from detections (count or area x constant)")
    ws = sp.add_subparsers(dest="weight_cmd", required=True)

    def common(p, predictions=True):
        p.add_argument("--workdir", default="runs/default")
        p.add_argument("--use", choices=["reviewed", "predictions"], default="reviewed",
                       help="boxes from the reviewed store or from predictions.json")
        if predictions:
            p.add_argument("--predictions", help="predictions.json (implies --use predictions)")
        p.add_argument("--reviewed-only", action="store_true", help="only photos marked reviewed")

    p = ws.add_parser("predict", help="per-photo count and estimated weight (+ error vs balance)")
    common(p)
    p.add_argument("--model", help="count_x_constant | area_x_constant (default: weight_config.json)")
    p.add_argument("--out", help="write CSV")
    p.set_defaults(fn=cmd_predict)
    p = ws.add_parser("calibrate", help="fit a model's constants on this project, save weight_config.json")
    common(p)
    p.add_argument("--model", default="count_x_constant", choices=["count_x_constant", "area_x_constant"])
    p.set_defaults(fn=cmd_calibrate)
