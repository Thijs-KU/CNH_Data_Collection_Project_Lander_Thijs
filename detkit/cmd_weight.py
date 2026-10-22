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


def cmd_features(a) -> int:
    """Every feature against the measured weight: correlation table (+ bootstrap interval) and plots. The plots
    work without weights too (then they show the value of each feature per photo)."""
    import numpy as np

    from .weight import SampleTable, feature_correlations, numeric_features

    allp = _samples(a).for_crop(a.crop)
    if not len(allp):
        print(f"no photos{f' of crop {a.crop!r}' if a.crop else ''}")
        return 1
    usable = SampleTable([s for s in allp if s.weight_g is not None and s.weight_g > 0])
    dropped = sum(1 for s in allp if s.weight_g is not None and s.weight_g <= 0)
    n_piles = len({s.meta.get("pile_id") or s.file for s in allp})
    print(f"{len(allp)} photos, {n_piles} piles, {len(usable)} with a weight > 0"
          + (f" ({dropped} with weight <= 0 left out)" if dropped else "")
          + (f", crop {a.crop}" if a.crop else ""))
    res = feature_correlations(usable, crop=a.crop, n_boot=a.boot) if len(usable) >= 3 else []
    feats = numeric_features(allp)
    if res:
        if a.boot:
            print(f"bootstrap resamples whole piles ({a.boot} rounds)")
        print(f"{'feature':24s} {'r':>7s} {'r2':>6s} {'slope':>12s}" + (f" {'95% CI of r':>17s}" if a.boot else ""))
        for e in res:
            ci = f" {_f(e['r_lo'], '{:+.3f}')} .. {_f(e['r_hi'], '{:+.3f}')}" if a.boot else ""
            print(f"{e['feature']:24s} {e['r']:+7.3f} {e['r2']:6.3f} {e['slope']:12.4g}{ci}")
        left = [k for k in feats if k not in {e["feature"] for e in res}]
        if left:
            print("no correlation (feature does not vary): " + ", ".join(left))
    else:
        print("no correlation: fewer than 3 photos with a weight > 0 (or the weights do not vary)")
    print(f"\nvalue of every feature over the {len(allp)} photos:")
    print(f"{'feature':24s} {'min':>11s} {'median':>11s} {'mean':>11s} {'p95':>11s} {'max':>11s} {'std':>10s}")
    for k in feats:
        v = np.array([s.get(k) for s in allp], dtype=float)
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        print(f"{k:24s} {v.min():11.5g} {np.median(v):11.5g} {v.mean():11.5g} {np.percentile(v, 95):11.5g} "
              f"{v.max():11.5g} {sd:10.4g}")
    if a.curve:
        _curve(a, usable, feats)
    if a.plot or a.plot_dir or a.show:
        return _plots(a, allp, usable, {e["feature"]: e for e in res}, feats)
    return 0


def _curve(a, usable, feats: list[str]) -> None:
    """``--curve FEATURE[,FEATURE...]|all``: how r and the slope settle as the number of photos grows. Prints a table
    per feature and always saves the figures (default folder ``<workdir>/reports/features``)."""
    import numpy as np

    from .weight import learning_curve

    names = feats if a.curve.strip().lower() == "all" else [k.strip() for k in a.curve.split(",") if k.strip()]
    bad = [k for k in names if k not in feats]
    if bad:
        print(f"--curve: not a numeric feature of every photo: {', '.join(bad)}; choose from: {', '.join(feats)}")
        names = [k for k in names if k in feats]
    if not names:
        return
    if len(usable) < 3:
        print("learning curve: needs at least 3 photos with a weight > 0")
        return
    n = len(usable)
    sizes = ([int(v) for v in a.sizes.split(",")] if a.sizes
             else sorted({int(round(v)) for v in np.linspace(3, n, min(8, n - 2))}))
    curves = {}
    for k in names:
        out = learning_curve(usable, k, sizes, crop=a.crop, repeats=a.repeats)
        if out is None or all(e["r_mean"] is None for e in out):
            print(f"\nlearning curve of {k}: no correlation (the feature does not vary)")
            continue
        curves[k] = out
        print(f"\nlearning curve of {k}: r and slope in {a.repeats} random subsets of whole piles per size")
        print(f"{'photos':>6s} {'piles':>6s} {'r mean':>8s} {'r 95% range':>17s} {'slope mean':>11s} {'slope 95% range':>25s}")
        for e in out:
            rr = f"{_f(e['r_lo'], '{:+.3f}')} .. {_f(e['r_hi'], '{:+.3f}')}"
            sr = f"{_f(e['slope_lo'], '{:.3g}')} .. {_f(e['slope_hi'], '{:.3g}')}"
            print(f"{e['n']:6d} {_f(e['n_piles'], '{:.1f}'):>6s} {_f(e['r_mean'], '{:+.3f}'):>8s} {rr:>17s} "
                  f"{_f(e['slope_mean'], '{:.3g}'):>11s} {sr:>25s}")
    if not curves:
        return
    try:
        from .weight import plots as WP
        plt = WP.pyplot(a.show)
    except ImportError:
        print("matplotlib is not installed: no curve plots (pip install matplotlib)")
        return
    if a.plot_dir:
        folder, stem = Path(a.plot_dir), "learning_curve_{}"
    elif a.plot:
        folder, stem = Path(a.plot).parent, Path(a.plot).stem + "_curve_{}"
    else:
        folder, stem = Path(a.workdir) / "reports" / "features", "learning_curve_{}"
    for k, out in curves.items():
        fig = WP.learning_curve_figure(plt, out, k, total=n, repeats=a.repeats)
        print(f"wrote {WP.save(fig, folder / (stem.format(k) + '.' + a.format))}")
        if not a.show:
            plt.close(fig)
    if len(curves) > 1:
        fig = WP.curves_overview_figure(plt, curves, total=n, repeats=a.repeats)
        print(f"wrote {WP.save(fig, folder / (stem.format('all') + '.' + a.format))}")
        if not a.show:
            plt.close(fig)


def _plots(a, allp, usable, by_feat: dict, feats: list[str]) -> int:
    """``--plot`` (one overview figure), ``--plot-dir`` (one figure per feature), ``--show`` (open the figures).
    Piles (photos with the same ``pile_id``) get their own colour and marker; see detkit/weight/plots.py."""
    try:
        from .weight import plots as WP
        plt = WP.pyplot(a.show)
    except ImportError:
        print("matplotlib is not installed: no plots (pip install matplotlib)")
        return 0
    styles = WP.pile_styles(allp, plt)
    if a.plot:
        fig = WP.overview_figure(plt, feats, allp, usable, by_feat, styles)
        print(f"wrote {WP.save(fig, a.plot)}")
        if not a.show:
            plt.close(fig)
    if a.plot_dir or (a.show and not a.plot):
        for k in feats:
            fig = WP.feature_figure(plt, k, allp, usable, by_feat.get(k), styles)
            if a.plot_dir:
                WP.save(fig, Path(a.plot_dir) / f"{k}.{a.format}")
            if not a.show:
                plt.close(fig)
        if a.plot_dir:
            print(f"wrote {len(feats)} plots to {a.plot_dir}")
    if a.show:
        plt.show()
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
    p.add_argument("--model", help="count_x_constant | area_x_constant | kernel_area_x_constant (default: weight_config.json)")
    p.add_argument("--out", help="write CSV")
    p.set_defaults(fn=cmd_predict)
    p = ws.add_parser("calibrate", help="fit a model's constants on this project, save weight_config.json")
    common(p)
    p.add_argument("--model", default="count_x_constant", choices=["count_x_constant", "area_x_constant", "kernel_area_x_constant"])
    p.set_defaults(fn=cmd_calibrate)
    p = ws.add_parser("features", help="correlation of every feature with the measured weight (task 2)")
    common(p)
    p.add_argument("--crop", help="only photos of this crop")
    p.add_argument("--boot", type=int, default=0, help="bootstrap rounds for the CI of r (0 = skip)")
    p.add_argument("--plot", help="one PNG with a scatter plot of weight against every feature")
    p.add_argument("--plot-dir", help="a folder with one larger PNG per feature")
    p.add_argument("--show", action="store_true", help="open the plots in windows (needs a display)")
    p.add_argument("--format", choices=["png", "pdf", "svg"], default="png",
                   help="file type of the per-feature and curve plots (default png; pdf / svg for the report)")
    p.add_argument("--curve", metavar="FEATURE", help="learning curve of this feature (several: comma separated; "
                   "all: every feature): how r and the slope settle as the number of photos grows. Always plotted: "
                   "in --plot-dir (or next to --plot), else in <workdir>/reports/features")
    p.add_argument("--sizes", help="photo counts for --curve, comma separated (default: 8 steps up to all photos)")
    p.add_argument("--repeats", type=int, default=200, help="random subsets per size for --curve")
    p.set_defaults(fn=cmd_features)
