"""Detector evaluation against COCO ground truth: P/R/F1, PR/AP, ROC, threshold sweep, COUNT and AREA error.

Ported (matching + curves) from vision-intern ``jobs/eval_pr_roc.py``.
Everything here is pure numpy on plain dicts (no torch), so it is cheap to test.

Matching (as upstream): per image and per class, predictions are visited by descending score; each
takes the unmatched GT box of the same class with the highest IoU >= ``iou``; else it is a false
positive. Unmatched GT boxes are false negatives. Because the visit order is by score, the matching
at any higher score threshold is a *subset* of the matching at the floor, so we match once and
threshold afterwards (same trick as COCO evaluation).

Classes are bridged by NAME (lower-case). GT classes the model cannot predict are dropped when the
model's label list is given (reported); prediction labels absent from the GT become extra classes
with 0 GT boxes (all their predictions are false positives).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from . import coco as C
from .merge import iou as box_iou

IOU_GRID = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def _norm(name: str) -> str:
    return str(name).strip().lower()


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


# ---- input normalisation ---------------------------------------------------------------------
def gt_from_coco(coco: dict) -> tuple[dict[str, list[tuple[list[float], str]]], list[str]]:
    """``({file: [(xyxy, class_name)]}, class_names)``; images without boxes map to []."""
    names = {c["id"]: _norm(c["name"]) for c in coco["categories"]}
    out: dict[str, list] = {i["file_name"]: [] for i in coco["images"]}
    fname = {i["id"]: i["file_name"] for i in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        out[fname[a["image_id"]]].append(([x, y, x + w, y + h], names[a["category_id"]]))
    classes = [names[c["id"]] for c in sorted(coco["categories"], key=lambda c: c["id"])]
    return out, classes


def preds_from_json(obj: dict) -> dict[str, list[tuple[list[float], float, str]]]:
    """Accept ``predict`` output ``{file: {detections:[{bbox,score,label}]}}`` (or ``{file: [dets]}``)."""
    out = {}
    for f, v in obj.items():
        dets = v["detections"] if isinstance(v, dict) else v
        out[f] = [(list(d["bbox"]), float(d["score"]), _norm(d.get("label", d.get("label_id", "?"))))
                  for d in dets]
    return out


# ---- matching --------------------------------------------------------------------------------
def match(gt: dict, preds: dict, classes: list[str], iou_thr: float = 0.5) -> dict:
    """Greedy per-image/per-class matching at the score floor. Returns rows + GT bookkeeping.

    ``rows``: one dict per prediction ``{file, cls, score, tp, iou, pbox, gbox}``.
    ``gt_rows``: one dict per GT box ``{file, cls, gbox, matched}``.
    """
    rows, gt_rows = [], []
    for f in sorted(gt):
        g = [(b, c) for b, c in gt[f] if c in classes]
        p = preds.get(f, [])
        used = [False] * len(g)
        gmatch = [None] * len(g)
        for pi in sorted(range(len(p)), key=lambda i: -p[i][1]):
            pb, ps, pc = p[pi]
            best, best_iou = -1, iou_thr
            for j, (gb, gc) in enumerate(g):
                if used[j] or gc != pc:
                    continue
                v = box_iou(pb, gb)
                if v >= best_iou:
                    best, best_iou = j, v
            if best >= 0:
                used[best] = True
            rows.append({"file": f, "cls": pc, "score": ps, "tp": best >= 0,
                         "iou": best_iou if best >= 0 else 0.0, "pbox": pb,
                         "gbox": g[best][0] if best >= 0 else None})
        for j, (gb, gc) in enumerate(g):
            gt_rows.append({"file": f, "cls": gc, "gbox": gb, "matched": used[j]})
    return {"rows": rows, "gt_rows": gt_rows, "files": sorted(gt), "iou": iou_thr}


# ---- curves and scalar metrics ---------------------------------------------------------------
def _safe_div(a, b):
    return a / b if b else 0.0


def prf(tp: int, fp: int, n_gt: int) -> dict:
    p, r = _safe_div(tp, tp + fp), _safe_div(tp, n_gt)
    return {"precision": p, "recall": r, "f1": _safe_div(2 * p * r, p + r),
            "tp": int(tp), "fp": int(fp), "fn": int(n_gt - tp), "n_gt": int(n_gt)}


def sweep(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> dict:
    """P/R/F1 at every distinct score used as ``>=`` threshold (descending). Empty-safe."""
    if len(scores) == 0:
        z = np.zeros(0)
        return {"thr": z, "precision": z, "recall": z, "f1": z, "tp": z, "fp": z}
    order = np.argsort(-scores, kind="stable")
    s, t = scores[order], tp[order].astype(float)
    ctp, cfp = np.cumsum(t), np.cumsum(1 - t)
    last = np.searchsorted(-s, -np.unique(s)[::-1], side="right") - 1   # last index per distinct score
    thr = s[last]
    ctp, cfp = ctp[last], cfp[last]
    prec = ctp / np.maximum(ctp + cfp, 1)
    rec = ctp / max(n_gt, 1) if n_gt else np.zeros_like(ctp)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return {"thr": thr, "precision": prec, "recall": rec, "f1": f1, "tp": ctp, "fp": cfp}


def average_precision(sw: dict) -> tuple[float, float]:
    """``(AP, AP_trapz)``. AP = VOC all-point interpolated area (monotone precision envelope);
    AP_trapz = upstream ``eval_pr_roc`` (trapezoid over the raw curve starting at (0, 1))."""
    if len(sw["thr"]) == 0:
        return 0.0, 0.0
    rec, prec = sw["recall"], sw["precision"]
    mrec = np.concatenate([[0.0], rec, [rec[-1]]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    env = np.maximum.accumulate(mpre[::-1])[::-1]
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * env[1:]))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    ap_t = float(trapz(np.concatenate([[1.0], prec]), np.concatenate([[0.0], rec])))
    return ap, ap_t


def roc_over_predictions(scores: np.ndarray, tp: np.ndarray) -> dict:
    """ROC where every *prediction* is a sample (TP=positive, FP=negative), score = confidence.

    There is no true-negative pool in detection, so this is NOT 1-specificity over all possible boxes:
    it measures how well the confidence separates real from false detections (calibration/ranking).
    AUC = P(random TP outranks random FP); NaN if either group is empty.
    """
    pos, neg = int(tp.sum()), int((~tp.astype(bool)).sum())
    if not pos or not neg:
        return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": float("nan")}
    order = np.argsort(-scores, kind="stable")
    s, t = scores[order], tp[order].astype(float)
    ctp, cfp = np.cumsum(t), np.cumsum(1 - t)
    keep = np.r_[np.diff(s) != 0, True]                       # one point per distinct score
    fpr = np.r_[0.0, cfp[keep] / neg]
    tpr = np.r_[0.0, ctp[keep] / pos]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(trapz(tpr, fpr))}


def _arrays(rows: list[dict]):
    return (np.array([r["score"] for r in rows], dtype=float),
            np.array([r["tp"] for r in rows], dtype=bool))


def count_metrics(pred: list[float], gt: list[float]) -> dict:
    """Agreement of per-image quantities (counts or areas). bias=mean(pred-gt); MAPE over gt>0;
    R^2 = 1 - SSres/SStot for the identity line pred=gt (can be negative); rel_total = sum(pred)/sum(gt)-1."""
    p, g = np.asarray(pred, float), np.asarray(gt, float)
    n = len(g)
    if n == 0:
        return {"n": 0}
    d = p - g
    nz = g > 0
    sst = float(((g - g.mean()) ** 2).sum())
    r = float(np.corrcoef(p, g)[0, 1]) if n > 1 and p.std() > 0 and g.std() > 0 else float("nan")
    return {"n": n, "bias": float(d.mean()), "mae": float(np.abs(d).mean()),
            "rmse": float(math.sqrt((d ** 2).mean())),
            "mape_pct": float(np.abs(d[nz] / g[nz]).mean() * 100) if nz.any() else float("nan"),
            "r2": float(1 - (d ** 2).sum() / sst) if sst > 0 and n > 1 else float("nan"),
            "pearson_r": r,
            "rel_total_err_pct": float((p.sum() / g.sum() - 1) * 100) if g.sum() > 0 else float("nan"),
            "sum_pred": float(p.sum()), "sum_gt": float(g.sum())}


# ---- per-threshold aggregates ----------------------------------------------------------------
def per_image_table(m: dict, thr: float) -> list[dict]:
    """Per-photo count/area comparison at score >= thr (all classes pooled)."""
    tab = {f: {"file": f, "gt_n": 0, "pred_n": 0, "tp": 0, "fp": 0, "fn": 0,
               "gt_area": 0.0, "pred_area": 0.0} for f in m["files"]}
    for g in m["gt_rows"]:
        t = tab[g["file"]]
        t["gt_n"] += 1
        t["gt_area"] += _area(g["gbox"])
    for r in m["rows"]:
        if r["score"] < thr:
            continue
        t = tab[r["file"]] if r["file"] in tab else None
        if t is None:
            continue
        t["pred_n"] += 1
        t["pred_area"] += _area(r["pbox"])
        t["tp" if r["tp"] else "fp"] += 1
    for t in tab.values():
        t["fn"] = t["gt_n"] - t["tp"]
        t["count_err"] = t["pred_n"] - t["gt_n"]
        t["count_err_pct"] = 100 * t["count_err"] / t["gt_n"] if t["gt_n"] else float("nan")
        t["area_err"] = t["pred_area"] - t["gt_area"]
        t["area_err_pct"] = 100 * t["area_err"] / t["gt_area"] if t["gt_area"] else float("nan")
    return [tab[f] for f in m["files"]]


def matched_table(m: dict) -> list[dict]:
    """One row per matched (TP at the floor) pred/GT pair, for size analysis; filter by ``score``."""
    out = []
    for r in m["rows"]:
        if not r["tp"]:
            continue
        pa, ga = _area(r["pbox"]), _area(r["gbox"])
        out.append({"file": r["file"], "class": r["cls"], "score": round(r["score"], 4),
                    "iou": round(r["iou"], 4),
                    "pred_box": [round(v, 1) for v in r["pbox"]], "gt_box": [round(v, 1) for v in r["gbox"]],
                    "pred_area": round(pa, 1), "gt_area": round(ga, 1),
                    "area_ratio": round(pa / ga, 4) if ga else float("nan"),
                    "w_ratio": round((r["pbox"][2] - r["pbox"][0]) / max(r["gbox"][2] - r["gbox"][0], 1e-9), 4),
                    "h_ratio": round((r["pbox"][3] - r["pbox"][1]) / max(r["gbox"][3] - r["gbox"][1], 1e-9), 4)})
    return out


def _rows_for(m: dict, cls: str | None):
    rows = [r for r in m["rows"] if cls is None or r["cls"] == cls]
    n_gt = sum(1 for g in m["gt_rows"] if cls is None or g["cls"] == cls)
    return rows, n_gt


def evaluate(gt_coco: dict, preds: dict, *, iou: float = 0.5, op_score: float = 0.5,
             model_labels: list[str] | None = None, map_grid: bool = False) -> dict:
    """Full evaluation. ``preds`` is the normalised dict from :func:`preds_from_json`.

    Returns a JSON-safe ``result`` (metrics) plus private ``_m`` / ``_sweeps`` for plotting/CSV.
    """
    gt, gt_classes = gt_from_coco(gt_coco)
    notes: list[str] = []
    dropped = {}
    classes = list(gt_classes)
    if model_labels is not None:
        ml = {_norm(x) for x in model_labels}
        dropped_c = [c for c in classes if c not in ml]
        for c in dropped_c:
            dropped[c] = sum(1 for v in gt.values() for _, k in v if k == c)
        classes = [c for c in classes if c in ml]
        if dropped:
            notes.append(f"GT classes the model cannot predict were dropped: {dropped}")
    extra = sorted({c for v in preds.values() for _, _, c in v} - set(classes) - set(dropped))
    if extra:
        classes += extra
        notes.append(f"prediction labels not in GT (all counted as false positives): {extra}")
    unknown_files = sorted(set(preds) - set(gt))
    if unknown_files:
        notes.append(f"{len(unknown_files)} prediction file(s) not in GT ignored, e.g. {unknown_files[:3]}")
    if gt and not (set(gt) & set(preds)):
        notes.append("WARNING: no prediction file name matches a GT image name; everything is FN/empty.")
    missing = sorted(set(gt) - set(preds))
    if missing and set(gt) & set(preds):
        notes.append(f"{len(missing)} GT image(s) have no prediction entry (treated as 0 detections): {missing[:3]}")
    preds = {f: [p for p in v if p[2] in classes or p[2] in extra] for f, v in preds.items() if f in gt}

    m = match(gt, preds, classes, iou)
    scores, tp = _arrays(m["rows"])
    n_gt = len(m["gt_rows"])
    sw_all = sweep(scores, tp, n_gt)
    ap, ap_t = average_precision(sw_all)

    def at(sw, t):
        k = int(np.sum(sw["thr"] >= t))
        ctp = int(sw["tp"][k - 1]) if k else 0
        cfp = int(sw["fp"][k - 1]) if k else 0
        return ctp, cfp

    def op_block(t, rows, ngt):
        s, x = _arrays(rows)
        keep = s >= t
        return prf(int(x[keep].sum()), int((~x[keep]).sum()), ngt)

    # best-F1 threshold (micro). Ties -> highest threshold (fewest false positives).
    if len(sw_all["thr"]):
        bi = int(len(sw_all["f1"]) - 1 - np.argmax(sw_all["f1"][::-1]))
        best = {"score": float(sw_all["thr"][bi]), **prf(int(sw_all["tp"][bi]), int(sw_all["fp"][bi]), n_gt)}
    else:
        best = {"score": None, **prf(0, 0, n_gt)}

    per_class, aps = {}, []
    for c in classes:
        rows, ngt = _rows_for(m, c)
        s, x = _arrays(rows)
        swc = sweep(s, x, ngt)
        apc, apc_t = average_precision(swc)
        bf = {"score": None}
        if len(swc["thr"]):
            bi = int(len(swc["f1"]) - 1 - np.argmax(swc["f1"][::-1]))
            bf = {"score": float(swc["thr"][bi]), "f1": float(swc["f1"][bi])}
        per_class[c] = {**op_block(op_score, rows, ngt), "ap": apc if ngt else float("nan"),
                        "n_pred_floor": len(rows), "best_f1_score": bf["score"], "best_f1": bf.get("f1")}
        if ngt:
            aps.append(apc)
    macro = {k: float(np.mean([v[k] for v in per_class.values() if v["n_gt"]])) if aps else float("nan")
             for k in ("precision", "recall", "f1")}
    macro["ap"] = float(np.mean(aps)) if aps else float("nan")

    result = {
        "iou": iou, "op_score": op_score, "n_images": len(gt), "n_gt": n_gt, "n_pred_floor": len(m["rows"]),
        "score_floor": float(scores.min()) if len(scores) else None,
        "classes": classes, "notes": notes, "dropped_gt_classes": dropped,
        "micro": {**op_block(op_score, m["rows"], n_gt), "ap": ap, "ap_trapz_upstream": ap_t},
        "macro": macro, "best_f1": best, "per_class": per_class,
        "roc": {"auc": roc_over_predictions(scores, tp)["auc"],
                "meaning": "AUC that a random true detection outranks a random false one by confidence; "
                           "no true negatives exist in detection, so this is not 1-specificity."},
    }
    if map_grid:
        by_iou = {}
        for t in IOU_GRID:
            mt = match(gt, preds, classes, t)
            aps_t = []
            for c in classes:
                rows, ngt = _rows_for(mt, c)
                if ngt:
                    aps_t.append(average_precision(sweep(*_arrays(rows), ngt))[0])
            by_iou[str(t)] = float(np.mean(aps_t)) if aps_t else float("nan")
        result["map_50_95"] = float(np.mean(list(by_iou.values())))
        result["ap_by_iou"] = by_iou

    # count / area error at the operating score and at the best-F1 score
    result["count_area"] = {}
    tables = {}
    for tag, t in (("op", op_score), ("best_f1", best["score"])):
        if t is None:
            continue
        tab = per_image_table(m, t)
        tables[tag] = tab
        result["count_area"][tag] = {
            "score": t,
            "count": count_metrics([r["pred_n"] for r in tab], [r["gt_n"] for r in tab]),
            "area": count_metrics([r["pred_area"] for r in tab], [r["gt_area"] for r in tab]),
            "worst_by_count": [{k: r[k] for k in ("file", "gt_n", "pred_n", "count_err", "count_err_pct")}
                               for r in sorted(tab, key=lambda r: -abs(r["count_err"]))[:5]],
            "worst_by_area": [{k: r[k] for k in ("file", "gt_area", "pred_area", "area_err_pct")}
                              for r in sorted(tab, key=lambda r: -abs(r["area_err"]))[:5]],
        }
        if len(classes) > 1:
            pc = {}
            for c in classes:
                mc = {"files": m["files"], "gt_rows": [g for g in m["gt_rows"] if g["cls"] == c],
                      "rows": [r for r in m["rows"] if r["cls"] == c]}
                tc = per_image_table(mc, t)
                pc[c] = count_metrics([r["pred_n"] for r in tc], [r["gt_n"] for r in tc])
            result["count_area"][tag]["per_class_count"] = pc
    mt = matched_table(m)
    ok = [r for r in mt if r["score"] >= op_score and not math.isnan(r["area_ratio"])]
    ar = np.array([r["area_ratio"] for r in ok]) if ok else np.zeros(0)
    result["matched"] = {
        "n_at_op": len(ok), "mean_iou": float(np.mean([r["iou"] for r in ok])) if ok else float("nan"),
        "area_ratio_median": float(np.median(ar)) if len(ar) else float("nan"),
        "area_ratio_mean": float(ar.mean()) if len(ar) else float("nan"),
        "area_ratio_p10_p90": [float(np.percentile(ar, 10)), float(np.percentile(ar, 90))] if len(ar) else None,
    }
    result["_m"], result["_sweep"], result["_tables"], result["_matched"] = m, sw_all, tables, mt
    result["_scores"], result["_tp"] = scores, tp
    result["_class_sweeps"] = {c: sweep(*_arrays(_rows_for(m, c)[0]), _rows_for(m, c)[1]) for c in classes}
    return result


# ---- outputs ---------------------------------------------------------------------------------
def _f(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0]) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _short(file: str) -> str:
    stem = Path(file).stem
    k = stem.rfind("_x")
    return stem[k + 1:] if k > 0 and "_y" in stem[k:] else stem[-9:]


def _plots(res: dict, out: Path, title: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    made = []
    sw = res["_sweep"]

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (c, s) in enumerate(res["_class_sweeps"].items()):
        if len(s["thr"]):
            ax.plot(s["recall"], s["precision"], color=cols[i % 6], lw=1.6,
                    label=f"{c} (AP={_f(res['per_class'][c]['ap'])})")
    if len(sw["thr"]) and len(res["classes"]) > 1:
        ax.plot(sw["recall"], sw["precision"], "k--", lw=1.2, label=f"micro (AP={_f(res['micro']['ap'])})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.02), ylim=(0, 1.02),
           title=f"PR curve @ IoU {res['iou']}\n{title}")
    ax.grid(alpha=0.3); ax.legend(loc="lower left")
    fig.savefig(out / "pr_curve.png", dpi=130, bbox_inches="tight"); plt.close(fig); made.append("pr_curve.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if len(sw["thr"]):
        ax.plot(sw["thr"], sw["precision"], color=cols[0], label="precision")
        ax.plot(sw["thr"], sw["recall"], color=cols[1], label="recall")
        ax.plot(sw["thr"], sw["f1"], color=cols[2], lw=2.2, label="F1")
        b = res["best_f1"]
        ax.axvline(b["score"], color="k", ls=":", label=f"best F1={_f(b['f1'])} @ score {_f(b['score'])}")
        ax.axvline(res["op_score"], color="grey", ls="--", lw=0.8, label=f"operating score {res['op_score']}")
    ax.set(xlabel="score threshold (keep predictions >= t)", ylim=(0, 1.02),
           title=f"P / R / F1 vs threshold @ IoU {res['iou']}\n{title}")
    ax.grid(alpha=0.3); ax.legend(loc="lower center", fontsize=8)
    fig.savefig(out / "f1_vs_threshold.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    made.append("f1_vs_threshold.png")

    roc = roc_over_predictions(res["_scores"], res["_tp"])
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(roc["fpr"], roc["tpr"], color=cols[0], label=f"AUC={_f(roc['auc'])}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="FPR: share of false detections kept", ylabel="TPR: share of true detections kept",
           title="ROC over predictions (no true negatives:\nconfidence separates real from false detections)")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(out / "roc.png", dpi=130, bbox_inches="tight"); plt.close(fig); made.append("roc.png")

    for key, unit in (("count", "count"), ("area", "box area (px^2)")):
        tab = res["_tables"].get("op")
        if not tab:
            continue
        x = [r["gt_n" if key == "count" else "gt_area"] for r in tab]
        y = [r["pred_n" if key == "count" else "pred_area"] for r in tab]
        cm = res["count_area"]["op"][key]
        fig, ax = plt.subplots(figsize=(5.4, 5))
        hi = max(x + y + [1]) * 1.05
        ax.plot([0, hi], [0, hi], "k--", lw=1, label="y = x")
        ax.scatter(x, y, color=cols[0], s=36, zorder=3)
        if len(x) <= 15:
            for r, xi, yi in zip(tab, x, y):
                ax.annotate(_short(r["file"]), (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set(xlabel=f"GT {unit}", ylabel=f"predicted {unit} (score >= {res['op_score']})",
               xlim=(0, hi), ylim=(0, hi),
               title=f"per-image {key}: bias {_f(cm.get('bias'), 1)}, MAE {_f(cm.get('mae'), 1)}, "
                     f"R2 {_f(cm.get('r2'), 3)}\n{title}")
        ax.set_title(ax.get_title(), fontsize=9); ax.grid(alpha=0.3); ax.legend(loc="upper left")
        fig.savefig(out / f"{key}_scatter.png", dpi=130, bbox_inches="tight"); plt.close(fig)
        made.append(f"{key}_scatter.png")
    return made


def _md(res: dict, meta: dict) -> str:
    mi, ma, b, ca = res["micro"], res["macro"], res["best_f1"], res["count_area"]
    L = [f"# Detection evaluation: {meta.get('name', '')}", "",
         f"- GT: `{meta.get('gt')}` ({meta.get('level')}-level, {res['n_images']} images, {res['n_gt']} GT boxes)",
         f"- Predictions: {meta.get('pred_desc')}", f"- IoU threshold {res['iou']}, operating score {res['op_score']}",
         f"- **GT caveat:** {meta.get('gt_note')}", ""]
    for n in res["notes"]:
        L.append(f"- NOTE: {n}")
    L += ["", "## Headline", "",
          "| | precision | recall | F1 | AP@IoU | TP | FP | FN |", "|---|---|---|---|---|---|---|---|",
          f"| micro @ score {res['op_score']} | {_f(mi['precision'])} | {_f(mi['recall'])} | {_f(mi['f1'])} | "
          f"{_f(mi['ap'])} | {mi['tp']} | {mi['fp']} | {mi['fn']} |",
          f"| best-F1 @ score {_f(b['score'])} | {_f(b['precision'])} | {_f(b['recall'])} | {_f(b['f1'])} | - | "
          f"{b['tp']} | {b['fp']} | {b['fn']} |",
          f"| macro over classes @ {res['op_score']} | {_f(ma['precision'])} | {_f(ma['recall'])} | {_f(ma['f1'])} | "
          f"{_f(ma['ap'])} | | | |", ""]
    if "map_50_95" in res:
        L += [f"mAP@[0.5:0.95] = **{_f(res['map_50_95'])}**; AP by IoU: " +
              ", ".join(f"{k}: {_f(v)}" for k, v in res["ap_by_iou"].items()), ""]
    L += [f"**Recommended operating score: {_f(b['score'])}** (maximises micro F1 on this GT; "
          "re-tune on your own reviewed data before trusting it).", "",
          f"ROC-style AUC over predictions = {_f(res['roc']['auc'])}. {res['roc']['meaning']}", "",
          "## Per class", "", "| class | GT | precision | recall | F1 | AP | best-F1 score |", "|---|---|---|---|---|---|---|"]
    for c, v in res["per_class"].items():
        L.append(f"| {c} | {v['n_gt']} | {_f(v['precision'])} | {_f(v['recall'])} | {_f(v['f1'])} | "
                 f"{_f(v['ap'])} | {_f(v['best_f1_score'])} |")
    for tag, label in (("op", f"operating score {res['op_score']}"), ("best_f1", f"best-F1 score {_f(b['score'])}")):
        if tag not in ca:
            continue
        c, a = ca[tag]["count"], ca[tag]["area"]
        L += ["", f"## Count and area error ({label})", "",
              "Per-image totals over all classes; bias = mean(pred - GT)."
              + (f" **Only {c['n']} images: R2/MAPE are not meaningful, read bias/MAE.**" if c["n"] < 5 else ""), "",
              "| quantity | n images | bias | MAE | RMSE | MAPE % | R2 | total error % |", "|---|---|---|---|---|---|---|---|",
              f"| count | {c['n']} | {_f(c.get('bias'), 2)} | {_f(c.get('mae'), 2)} | {_f(c.get('rmse'), 2)} | "
              f"{_f(c.get('mape_pct'), 1)} | {_f(c.get('r2'))} | {_f(c.get('rel_total_err_pct'), 1)} |",
              f"| box area px^2 | {a['n']} | {_f(a.get('bias'), 0)} | {_f(a.get('mae'), 0)} | {_f(a.get('rmse'), 0)} | "
              f"{_f(a.get('mape_pct'), 1)} | {_f(a.get('r2'))} | {_f(a.get('rel_total_err_pct'), 1)} |", "",
              "Worst images by |count error|: " + "; ".join(
                  f"{w['file']} (GT {w['gt_n']}, pred {w['pred_n']}, {_f(w['count_err_pct'], 0)}%)"
                  for w in ca[tag]["worst_by_count"]), ""]
    L += ["## Per-photo table (operating score)", "",
          "| file | GT | pred | TP | FP | FN | count err % | GT area | pred area | area err % |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["_tables"].get("op", []):
        L.append(f"| {r['file']} | {r['gt_n']} | {r['pred_n']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                 f"{_f(r['count_err_pct'], 1)} | {r['gt_area']:.0f} | {r['pred_area']:.0f} | {_f(r['area_err_pct'], 1)} |")
    mt = res["matched"]
    L += ["", "## Matched boxes (score >= operating score)", "",
          f"{mt['n_at_op']} matched pairs; mean IoU {_f(mt['mean_iou'])}; pred/GT area ratio median "
          f"{_f(mt['area_ratio_median'])}, mean {_f(mt['area_ratio_mean'])}, p10-p90 {mt['area_ratio_p10_p90'] and [round(v, 3) for v in mt['area_ratio_p10_p90']]}. "
          "A ratio far from 1 means the boxes are systematically too large/small, which biases pixels -> weight "
          "even when the count is right. Full table: `matched_boxes.csv`.", "",
          "## Files", "", "`eval.json`, `per_image.csv`, `per_class.csv`, `threshold_sweep.csv`, `pr_curve.csv`, "
          "`matched_boxes.csv`, `pr_curve.png`, `f1_vs_threshold.png`, `roc.png`, `count_scatter.png`, `area_scatter.png`."]
    return "\n".join(L) + "\n"


def write_report(res: dict, out_dir: str | Path, meta: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    public = {k: v for k, v in res.items() if not k.startswith("_")}
    public["meta"] = meta
    (out / "eval.json").write_text(json.dumps(public, indent=1, default=float), encoding="utf-8")
    for tag, tab in res["_tables"].items():
        _write_csv(out / ("per_image.csv" if tag == "op" else f"per_image_{tag}.csv"), tab)
    _write_csv(out / "per_class.csv", [{"class": c, **{k: v for k, v in d.items()}} for c, d in res["per_class"].items()])
    sw = res["_sweep"]
    _write_csv(out / "threshold_sweep.csv",
               [{"score": float(t), "precision": float(p), "recall": float(r), "f1": float(f), "tp": int(a), "fp": int(b)}
                for t, p, r, f, a, b in zip(sw["thr"], sw["precision"], sw["recall"], sw["f1"], sw["tp"], sw["fp"])],
               ["score", "precision", "recall", "f1", "tp", "fp"])
    _write_csv(out / "pr_curve.csv", [{"recall": float(r), "precision": float(p), "score": float(t)}
                                      for t, p, r in zip(sw["thr"], sw["precision"], sw["recall"])],
               ["recall", "precision", "score"])
    _write_csv(out / "matched_boxes.csv", res["_matched"],
               ["file", "class", "score", "iou", "pred_box", "gt_box", "pred_area", "gt_area", "area_ratio", "w_ratio", "h_ratio"])
    _plots(res, out, meta.get("name", ""))
    (out / "eval.md").write_text(_md(res, meta), encoding="utf-8")
    return out
