import json
import math

import pytest

from detkit import coco as C
from detkit.cmd_eval import cmd_eval
from detkit.evaluate import (average_precision, count_metrics, evaluate, match, preds_from_json,
                             roc_over_predictions, sweep, write_report)


def mk_gt(items, classes=("a",)):
    """items: {file: [(xyxy, cls_name)]} -> COCO."""
    c = C.new_coco(list(classes))
    for f, boxes in items.items():
        i = C.add_image(c, f, 1000, 1000)
        for b, n in boxes:
            C.add_box(c, i, list(classes).index(n), b)
    return c


def dets(*ds):
    return [{"bbox": b, "score": s, "label": l} for b, s, l in ds]


G1, G2, FAR = [0, 0, 10, 10], [50, 50, 60, 60], [200, 200, 210, 210]


def base():
    gt = mk_gt({"i.jpg": [(G1, "a"), (G2, "a")]})
    pj = {"i.jpg": {"detections": dets((G1, 0.9, "a"), (FAR, 0.8, "a"), (G2, 0.7, "a"))}}
    return gt, preds_from_json(pj)


def test_hand_computed_prf_ap_and_best_threshold():
    gt, pr = base()
    r = evaluate(gt, pr, iou=0.5, op_score=0.0)
    m = r["micro"]
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 0)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == 1 and m["f1"] == pytest.approx(0.8)
    assert m["ap"] == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))     # envelope AP
    r = evaluate(gt, pr, op_score=0.85)
    assert (r["micro"]["tp"], r["micro"]["fp"], r["micro"]["fn"]) == (1, 0, 1)
    assert r["micro"]["f1"] == pytest.approx(2 / 3)
    b = r["best_f1"]
    assert b["score"] == pytest.approx(0.7) and b["f1"] == pytest.approx(0.8)


def test_sweep_thresholds_and_ties():
    import numpy as np
    sw = sweep(np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), 2)
    assert list(sw["thr"]) == [0.9, 0.8, 0.7]
    assert list(sw["f1"]) == pytest.approx([2 / 3, 0.5, 0.8])
    sw = sweep(np.array([0.5, 0.5, 0.3]), np.array([True, False, False]), 1)   # tie handled as one threshold
    assert list(sw["thr"]) == [0.5, 0.3] and list(sw["tp"]) == [1, 1] and list(sw["fp"]) == [1, 2]
    ap, _ = average_precision(sw)
    assert ap == pytest.approx(0.5)      # recall 1.0 reached at precision 0.5 (best after envelope)


def test_greedy_matching_and_iou_threshold():
    gt = {"i": [([0, 0, 10, 10], "a")]}
    # higher-score pred with lower IoU wins the GT (greedy by score); the better-IoU one is FP
    p = {"i": [([0, 0, 10, 7], 0.9, "a"), ([0, 0, 10, 10], 0.8, "a")]}
    m = match(gt, p, ["a"], 0.5)
    assert [r["tp"] for r in m["rows"]] == [True, False] and m["rows"][0]["iou"] == pytest.approx(0.7)
    # IoU 4/... box: inter 40, union 160 -> 0.25
    p2 = {"i": [([0, 0, 10, 4], 0.9, "a")]}
    assert not match(gt, p2, ["a"], 0.5)["rows"][0]["tp"]
    assert match(gt, p2, ["a"], 0.3)["rows"][0]["tp"]              # IoU 0.4 passes a 0.3 threshold


def test_class_handling_micro_macro_and_labels():
    gt = mk_gt({"i.jpg": [(G1, "a"), (G2, "b")]}, classes=("a", "b"))
    pr = preds_from_json({"i.jpg": {"detections": dets((G1, 0.9, "a"), (G2, 0.9, "a"), (FAR, 0.6, "Z"))}})
    r = evaluate(gt, pr, op_score=0.0)
    a, b = r["per_class"]["a"], r["per_class"]["b"]
    assert (a["tp"], a["fp"], a["fn"]) == (1, 1, 0)                # b-box predicted as a is an FP for a
    assert (b["tp"], b["fp"], b["fn"]) == (0, 0, 1)
    assert r["per_class"]["z"]["n_gt"] == 0 and r["per_class"]["z"]["fp"] == 1
    assert any("not in GT" in n for n in r["notes"])
    assert r["micro"]["tp"] == 1 and r["micro"]["fp"] == 2 and r["micro"]["fn"] == 1
    assert r["macro"]["recall"] == pytest.approx(0.5)              # (1 + 0) / 2, z (no GT) excluded
    # model that cannot predict b: b GT dropped and reported
    r2 = evaluate(gt, pr, op_score=0.0, model_labels=["a"])
    assert r2["dropped_gt_classes"] == {"b": 1} and r2["n_gt"] == 1


def test_empty_cases():
    gt = mk_gt({"i.jpg": [(G1, "a")], "e.jpg": []})
    r = evaluate(gt, {})                                           # no predictions at all
    assert r["micro"]["recall"] == 0 and r["micro"]["fn"] == 1 and r["best_f1"]["score"] is None
    assert any("no prediction file name matches" in n for n in r["notes"])
    r = evaluate(mk_gt({"e.jpg": []}), preds_from_json({"e.jpg": {"detections": dets((G1, 0.9, "a"))}}))
    assert r["micro"]["fp"] == 1 and r["micro"]["precision"] == 0 and r["micro"]["n_gt"] == 0
    assert math.isnan(r["per_class"]["a"]["ap"])
    r = evaluate(mk_gt({"e.jpg": []}), {"e.jpg": []})              # nothing anywhere
    assert r["micro"]["f1"] == 0 and r["count_area"]["op"]["count"]["bias"] == 0


def test_count_and_area_metrics():
    m = count_metrics([1, 2, 3], [1, 2, 4])
    assert m["bias"] == pytest.approx(-1 / 3) and m["mae"] == pytest.approx(1 / 3)
    assert m["mape_pct"] == pytest.approx(100 * (0.25 / 3)) and m["rel_total_err_pct"] == pytest.approx(-100 / 7)
    assert m["r2"] == pytest.approx(1 - 1 / (14 / 3))              # SStot of [1,2,4] = 4.667
    m = count_metrics([2, 4], [3, 3])
    assert m["bias"] == 0 and m["mae"] == 1 and m["mape_pct"] == pytest.approx(100 / 3) and math.isnan(m["r2"])
    assert count_metrics([], []) == {"n": 0}

    gt = mk_gt({"i.jpg": [(G1, "a"), (G2, "a")], "j.jpg": [(G1, "a")]})
    pr = preds_from_json({"i.jpg": {"detections": dets(([0, 0, 10, 10], 0.9, "a"))},
                          "j.jpg": {"detections": dets(([0, 0, 20, 10], 0.9, "a"), (FAR, 0.9, "a"))}})
    r = evaluate(gt, pr, op_score=0.5)
    tab = {t["file"]: t for t in r["_tables"]["op"]}
    assert tab["i.jpg"]["count_err"] == -1 and tab["j.jpg"]["count_err"] == 1
    assert tab["j.jpg"]["pred_area"] == 300 and tab["j.jpg"]["area_err_pct"] == pytest.approx(200.0)
    assert r["count_area"]["op"]["count"]["bias"] == 0
    assert r["count_area"]["op"]["worst_by_count"][0]["count_err"] in (-1, 1)


def test_matched_table_area_ratio_and_roc():
    gt = mk_gt({"i.jpg": [(G1, "a")]})
    pr = preds_from_json({"i.jpg": {"detections": dets(([0, 0, 10, 8], 0.9, "a"), (FAR, 0.3, "a"))}})
    r = evaluate(gt, pr)
    (row,) = r["_matched"]
    assert row["area_ratio"] == 0.8 and row["iou"] == 0.8 and row["gt_box"] == G1
    assert r["matched"]["area_ratio_median"] == pytest.approx(0.8)
    roc = roc_over_predictions(r["_scores"], r["_tp"])
    assert roc["auc"] == 1.0                                       # the TP outranks the FP
    import numpy as np
    assert math.isnan(roc_over_predictions(np.array([0.5]), np.array([True]))["auc"])


def test_map_grid():
    gt, pr = base()
    r = evaluate(gt, pr, map_grid=True)
    assert r["ap_by_iou"]["0.5"] == pytest.approx(r["micro"]["ap"]) and r["ap_by_iou"]["0.95"] == pytest.approx(r["micro"]["ap"])
    assert 0 < r["map_50_95"] <= 1


def test_report_files_and_cli_with_predictions(tmp_path, capsys):
    from detkit.project import Project

    wd = tmp_path / "w"
    Project(workdir=wd, classes=["a"]).save()
    gt, pr = base()
    C.save(gt, wd / "annotations" / "reviewed_coco.json")
    (tmp_path / "p.json").write_text(json.dumps({"i.jpg": {"detections": dets((G1, 0.9, "a"), (FAR, 0.8, "a"), (G2, 0.7, "a"))}}))

    class A: pass
    a = A()
    a.__dict__.update(workdir=str(wd), gt="reviewed", pred=str(tmp_path / "p.json"), model=None, images=None,
                      iou=0.5, op_score=0.5, map=True, level="auto", tile=None, overlap=None, crop=None,
                      gt_note=None, name="t", out=str(tmp_path / "out"), score=0.05)
    assert cmd_eval(a) == 0
    out = tmp_path / "out"
    for f in ("eval.json", "eval.md", "per_image.csv", "per_class.csv", "threshold_sweep.csv", "pr_curve.csv",
              "matched_boxes.csv", "pr_curve.png", "f1_vs_threshold.png", "roc.png", "count_scatter.png"):
        assert (out / f).stat().st_size > 0, f
    j = json.loads((out / "eval.json").read_text())
    assert j["best_f1"]["score"] == pytest.approx(0.7) and "map_50_95" in j
    assert "Recommended operating score" in (out / "eval.md").read_text()
