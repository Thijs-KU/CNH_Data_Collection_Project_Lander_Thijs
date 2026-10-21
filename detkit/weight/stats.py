"""Statistics for judging weight models (numpy only)."""

from __future__ import annotations

import numpy as np

from .features import SampleTable


def evaluate_model(model, samples: SampleTable) -> dict:
    """Score ``model.predict`` against the true weights: n, mae_g, mape, bias_g (mean pred - true), r2,
    per-crop breakdown (same metrics) and ``rows`` (file, crop, true_g, pred_g, error_g)."""
    rows = []
    for s in samples.with_weight():
        est = model.predict(s.features, s.crop)
        rows.append({"file": s.file, "crop": s.crop, "true_g": s.weight_g, "pred_g": est.weight_g,
                     "error_g": est.weight_g - s.weight_g})
    res = _metrics(rows)
    res["per_crop"] = {c: _metrics([r for r in rows if r["crop"] == c])
                       for c in sorted({r["crop"] or "?" for r in rows})}
    res["rows"] = rows
    return res


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0, "mae_g": None, "mape": None, "bias_g": None, "r2": None}
    err = np.array([r["error_g"] for r in rows])
    true = np.array([r["true_g"] for r in rows])
    nz = true != 0
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {"n": n, "mae_g": float(np.abs(err).mean()), "bias_g": float(err.mean()),
            "mape": float(np.mean(np.abs(err[nz]) / true[nz])) if nz.any() else None,
            "r2": 1 - float(np.sum(err ** 2)) / ss_tot if n > 1 and ss_tot > 0 else None}


def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)) and np.isfinite(v)


def numeric_features(samples: SampleTable) -> list[str]:
    """Feature keys that every photo of the table has as a number (so a ``count_<label>`` that only some photos
    have is not in the list)."""
    rows = list(samples)
    return [k for k in dict.fromkeys(k for s in rows for k in s.features)
            if all(_is_num(s.features.get(k)) for s in rows)]


def _xy(samples: SampleTable, x_key: str, y_key: str = "weight_g", crop: str | None = None):
    """``x``, ``y`` and the pile of every weighed photo (of ``crop``) that has a numeric ``x_key``.

    The pile is the photo's ``pile_id``; a photo without one counts as a pile of its own. Photos of one pile are
    not independent, so the functions below resample / subsample whole piles, never single photos."""
    xs, ys, piles = [], [], []
    for s in samples.with_weight():
        if crop is not None and s.crop != crop:
            continue
        x, y = s.get(x_key), s.get(y_key)
        if not (_is_num(x) and _is_num(y)):
            continue
        xs.append(float(x))
        ys.append(float(y))
        pid = s.meta.get("pile_id")
        piles.append(f"pile:{pid}" if pid not in (None, "") else f"photo:{s.file}")
    return np.array(xs), np.array(ys), np.array(piles, dtype=object)


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """``(r, slope, intercept)`` of the least-squares line y = slope * x + intercept; all NaN when it is undefined
    (fewer than 3 points, or x or y constant, also up to rounding error)."""
    if len(x) < 3 or any(np.ptp(v) <= 1e-9 * max(1.0, float(np.abs(v).max())) for v in (x, y)):
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.corrcoef(x, y)[0, 1]), float(slope), float(intercept)


def _pile_members(piles: np.ndarray) -> list[np.ndarray]:
    """Row indices of each pile."""
    _, inv = np.unique(piles, return_inverse=True)
    return [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]


def feature_correlations(samples: SampleTable, y_key: str = "weight_g", crop: str | None = None,
                         features: list[str] | None = None, n_boot: int = 0, seed: int = 0) -> list[dict]:
    """How well does each feature follow the measured weight? One dict per feature, best first (by |r|):
    ``feature``, ``n`` (photos), ``r`` (Pearson), ``r2``, ``slope`` / ``intercept`` of the fitted line and, with
    ``n_boot`` > 0, ``r_lo`` / ``r_hi`` (pile-wise bootstrap interval of r, see :func:`bootstrap_ci`).

    ``features`` defaults to every numeric feature that all weighed photos of the crop have (so ``count_wheat``
    is only there for a wheat-only table); features that never change (r undefined) are left out."""
    rows = [s for s in samples.with_weight() if crop is None or s.crop == crop]
    if features is None:
        features = numeric_features(SampleTable(rows))
    out = []
    for k in features:
        x, y, _ = _xy(samples, k, y_key, crop)
        r, slope, intercept = _fit(x, y)
        if np.isnan(r):
            continue
        e = {"feature": k, "n": len(x), "r": r, "r2": r * r, "slope": slope, "intercept": intercept}
        if n_boot:
            ci = bootstrap_ci(samples, k, y_key, crop, "r", n_boot, seed=seed)
            e["r_lo"], e["r_hi"] = ci if ci else (None, None)
        out.append(e)
    return sorted(out, key=lambda e: -abs(e["r"]))


def learning_curve(samples: SampleTable, x_key: str, sizes: list[int], y_key: str = "weight_g",
                   crop: str | None = None, repeats: int = 200, seed: int = 0):
    """How does the relation between ``x_key`` and ``y_key`` settle as the number of samples grows?

    For every size ``n`` in ``sizes``: ``repeats`` times draw a random subset of ``n`` weighed photos, made of
    whole piles (piles in random order, the last one cut to fit; photos of one pile are not independent), and fit
    y on x. One dict per size, in the order of ``sizes``: ``n``, ``n_piles`` (mean piles in a subset), and for
    ``r`` and ``slope`` the mean, std and the central 95 % range over the repeats (``r_mean``, ``r_std``,
    ``r_lo``, ``r_hi``, ``slope_mean`` ...). The curve has settled when that range stops shrinking. At the full
    size every repeat uses all the data, so the spread there is 0 by construction; a size above the number of
    weighed photos gets ``None`` values. Returns ``None`` with fewer than 3 weighed photos."""
    x, y, piles = _xy(samples, x_key, y_key, crop)
    if len(x) < 3:
        return None
    members = _pile_members(piles)
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        entry: dict = {"n": int(n), "n_piles": None}
        rs, slopes, npiles = [], [], []
        if 3 <= n <= len(x):
            for _ in range(repeats):
                idx, used = [], 0
                for k in rng.permutation(len(members)):
                    idx.extend(members[k])
                    used += 1
                    if len(idx) >= n:
                        break
                idx = np.array(idx[:n])
                r, slope, _ = _fit(x[idx], y[idx])
                rs.append(r)
                slopes.append(slope)
                npiles.append(used)
        for name, vals in (("r", rs), ("slope", slopes)):
            v = np.array(vals, dtype=float)
            v = v[~np.isnan(v)]
            ok = len(v) > 0
            entry[f"{name}_mean"] = float(v.mean()) if ok else None
            entry[f"{name}_std"] = float(v.std(ddof=1)) if len(v) > 1 else (0.0 if ok else None)
            entry[f"{name}_lo"] = float(np.percentile(v, 2.5)) if ok else None
            entry[f"{name}_hi"] = float(np.percentile(v, 97.5)) if ok else None
        if npiles:
            entry["n_piles"] = float(np.mean(npiles))
        out.append(entry)
    return out


def bootstrap_ci(samples: SampleTable, x_key: str, y_key: str = "weight_g", crop: str | None = None,
                 stat: str = "r", n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Confidence interval ``(lo, hi)`` of the correlation (``stat="r"``) or of the slope (``stat="slope"``,
    least squares with intercept) of ``y_key`` against ``x_key``: the percentile bootstrap.

    Each of the ``n_boot`` rounds draws piles with replacement (as many as there are piles) and takes all their
    photos (cluster bootstrap), because photos of one pile (``pile_id``) are not independent samples; resampling
    single photos would make the interval too narrow. ``alpha`` = 0.05 gives a 95 % interval. Returns ``None``
    with fewer than 3 weighed photos or fewer than 2 piles."""
    if stat not in ("r", "slope"):
        raise ValueError(f"stat must be 'r' or 'slope', got {stat!r}")
    x, y, piles = _xy(samples, x_key, y_key, crop)
    if len(x) < 3:
        return None
    members = _pile_members(piles)
    if len(members) < 2:
        return None
    rng = np.random.default_rng(seed)
    which = 0 if stat == "r" else 1
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        vals.append(_fit(x[idx], y[idx])[which])
    v = np.array(vals)
    v = v[~np.isnan(v)]
    if not len(v):
        return None
    return float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))


def propagate(rel_errors: dict[str, float], weight_g: float) -> dict:
    """TODO(student) task 3. ``rel_errors`` = {stage: relative error of that stage}, e.g.
    {"scale": 0.02, "detector": 0.05, "kernel_weight": 0.03}, measured in your own experiments.

    Return a dict with ``rel_total`` (relative error of the weight), ``weight_u_g`` (absolute, in g),
    ``dominant`` (stage contributing most) and ``budget`` ({stage: share of the total})."""
    raise NotImplementedError("propagate(): student task 3 (docs/STUDENT_TASKS.md)")
