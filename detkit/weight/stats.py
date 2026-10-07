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


def learning_curve(samples: SampleTable, x_key: str, sizes: list[int], y_key: str = "weight_g",
                   crop: str | None = None, repeats: int = 200):
    """TODO(student) task 2. How does the relation between ``x_key`` and ``y_key`` settle as the number of
    samples grows? Return one dict per value in ``sizes``, each with at least ``"n"`` (the size) plus what
    you measured over ``repeats`` random subsets of that size. Photos of the same pile (``pile_id``) are
    not independent samples. Return ``None`` with fewer than 3 weighed photos."""
    raise NotImplementedError("learning_curve: student task 2 (docs/STUDENT_TASKS.md)")


def bootstrap_ci(samples: SampleTable, x_key: str, y_key: str = "weight_g", crop: str | None = None,
                 stat: str = "r", n_boot: int = 1000):
    """TODO(student) task 2. Confidence interval of the correlation (``stat="r"``) or of the slope
    (``stat="slope"``) of ``y_key`` against ``x_key``; return ``(lo, hi)``. Photos of the same pile
    (``pile_id``) are not independent samples. Return ``None`` with fewer than 3 weighed photos."""
    raise NotImplementedError("bootstrap_ci: student task 2 (docs/STUDENT_TASKS.md)")


def propagate(rel_errors: dict[str, float], weight_g: float) -> dict:
    """TODO(student) task 3. ``rel_errors`` = {stage: relative error of that stage}, e.g.
    {"scale": 0.02, "detector": 0.05, "kernel_weight": 0.03}, measured in your own experiments.

    Return a dict with ``rel_total`` (relative error of the weight), ``weight_u_g`` (absolute, in g),
    ``dominant`` (stage contributing most) and ``budget`` ({stage: share of the total})."""
    raise NotImplementedError("propagate(): student task 3 (docs/STUDENT_TASKS.md)")
