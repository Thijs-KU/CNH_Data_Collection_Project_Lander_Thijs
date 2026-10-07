"""Self-check for the student tasks (docs/STUDENT_TASKS.md).

Run:  python -m pytest tests/test_student_tasks.py -q -rs

A task that is not implemented yet is reported as SKIPPED with its task number ("TODO task 2").
Only tasks 2 and 3 have code; the others are experiments and the report (docs/STUDENT_TASKS.md).
Once you implement it, the test below runs for real and must pass. Do not edit these tests to make them pass.
"""

import numpy as np
import pytest

from detkit.weight import (AreaTimesConstant, CountTimesConstant, SampleTable, bootstrap_ci, learning_curve,
                           propagate)

A2G = 0.002          # true grams per px^2 of total box area
MM = 0.5             # mm per px used for the synthetic scale


def synthetic(n=30, seed=0, crop="wheat"):
    rng = np.random.default_rng(seed)
    t = SampleTable()
    for i in range(n):
        k = int(rng.integers(3, 40))                       # grains in the photo
        side = float(rng.uniform(18, 26))
        boxes = [{"bbox": [j * 30, 0, j * 30 + side, side], "label": crop, "score": 0.9} for j in range(k)]
        area = k * side * side
        w = A2G * area * (1 + rng.normal(0, 0.03))
        t.add(f"{crop}{i}.jpg", boxes, {"crop": crop, "total_weight_g": w, "scale_mm_per_px": MM,
                                        "pile_id": f"p{i // 2}"})
    return t


def todo(task, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except NotImplementedError:
        pytest.skip(f"TODO task {task}: not implemented yet (docs/STUDENT_TASKS.md)")


def test_task2_learning_curve_one_entry_per_size():
    out = todo(2, learning_curve, synthetic(40), "total_area_px", sizes=[5, 10, 20, 40], repeats=50)
    assert out is not None and len(out) == 4 and [e["n"] for e in out] == [5, 10, 20, 40]


def test_task2_bootstrap_ci_brackets_r():
    lo, hi = todo(2, bootstrap_ci, synthetic(40), "total_area_px", stat="r", n_boot=200)
    assert 0.9 < lo <= hi <= 1.0


def test_task2_too_few_samples_give_none():
    assert todo(2, bootstrap_ci, synthetic(2), "total_area_px", n_boot=50) is None
    assert todo(2, learning_curve, synthetic(2), "total_area_px", sizes=[2], repeats=5) is None


def test_task3_propagate_returns_budget():
    out = todo(3, propagate, {"scale": 0.02, "detector": 0.05, "kernel_weight": 0.03}, weight_g=100.0)
    assert out["dominant"] == "detector" and out["weight_u_g"] == pytest.approx(100.0 * out["rel_total"])
    assert 0.05 <= out["rel_total"] <= 0.02 + 0.05 + 0.03
    assert sum(out["budget"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("model", [CountTimesConstant, AreaTimesConstant])
def test_task3_models_give_bounds(model):
    t = synthetic()
    m = model().fit(t)
    est = m.predict(t.rows[0].features, "wheat")
    if est.low_g is None or est.high_g is None:
        pytest.skip(f"TODO task 3: {model.__name__}.predict should return low_g/high_g")
    assert est.low_g < est.weight_g < est.high_g
