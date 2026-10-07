"""Inference tab logic (no Gradio): detect -> features -> weight, save as samples, model constants.

Everything heavy is reached through module attributes (``P.load_bundle``, ``P.propose_rfdetr``) so tests
can swap in a fake predictor. Boxes are in ORIGINAL photo pixels.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Callable

from PIL import Image

from .. import weight as W
from ..train import META
from ..viz import draw_boxes
from . import logic as L
from . import proposals as P

RESULT_COLUMNS = ["file", "counts", "count", "total_area_px", "weight_g", "low_g", "high_g",
                  "measured_g", "error_pct", "note"]
CONST_COLUMNS = ["crop", "g_per_kernel", "g_per_px2"]     # count_x_constant, area_x_constant
SAMPLE_DIR = "models"                       # W/models/<name>.json = fitted student model (optional)


# ---- defaults --------------------------------------------------------------------------------------
def default_score(workdir: Path, fallback: float = 0.5) -> tuple[float, str]:
    """Best-F1 score of the newest ``W/reports/eval_*/eval.json``, else *fallback*."""
    for p in sorted(Path(workdir).glob("reports/eval_*/eval.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))["best_f1"]["score"]
        except (ValueError, KeyError, TypeError):
            continue
        if s:
            return round(float(s), 3), f"best-F1 score from {p.parent.name}"
    return fallback, "no eval report found: default 0.5 (run `detkit eval` to get the best-F1 score)"


def ckpt_defaults(ckpt: str | None, project) -> dict:
    """tile / overlap from the checkpoint's detkit_meta.json, else the project's."""
    d = {"tile": project.tile, "overlap": project.overlap}
    if ckpt:
        f = Path(ckpt) / META
        if f.exists():
            try:
                tp = json.loads(f.read_text(encoding="utf-8")).get("tile_params", {})
            except ValueError:
                tp = {}
            d.update({k: tp[k] for k in ("tile", "overlap") if tp.get(k) is not None})
    return d


def model_choices() -> list[tuple[str, str]]:
    return [(f"{n}{'' if m.implemented else '  (student TODO)'}", n) for n, m in W.MODELS.items()]


def load_weight_model(workdir: Path, name: str, cfg: dict) -> W.WeightModel:
    """Model by name; feature x constant models use the (editable) constants of the config; other models
    are restored from ``W/models/<name>.json`` when a student saved one."""
    key = getattr(W.MODELS.get(name), "config_key", None)
    if key:
        return W.get_model(name, constants=cfg[key])
    f = Path(workdir) / SAMPLE_DIR / f"{name}.json"
    return W.WeightModel.load(f) if f.exists() else W.get_model(name)


def todo_message(name: str) -> str:
    return f"{name}: student TODO, see docs/STUDENT_TASKS.md"


# ---- one photo -------------------------------------------------------------------------------------
def dets_from_boxes(boxes: list[dict]) -> list[dict]:
    return [{"bbox": b["bbox"], "score": b.get("score"), "label": b["class"]} for b in boxes]


def detect_photo(path: Path, bundle, project, *, crop: str | None, score: float, tile: int, overlap: float,
                 max_side: int = 0, roi_text: str = "") -> list[dict]:
    """Tiled RF-DETR detections for one photo as ``[{bbox, score, label}]`` (project class names)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        roi = P.parse_roi(roi_text, im.size)
        boxes, _info = P.propose_rfdetr(im, bundle, project.classes, crop or None, score=score, tile=tile,
                                        overlap=overlap, max_side=max_side, roi=roi)
    return dets_from_boxes(boxes)


def dominant_class(dets: list[dict]) -> str | None:
    labels = [d["label"] for d in dets]
    return max(set(labels), key=labels.count) if labels else None


def estimate_row(file: str, dets: list[dict], meta: dict, model: W.WeightModel, cfg: dict) -> dict:
    """One results-table row: features -> model -> weight (+ error vs the measured weight)."""
    feats = W.image_features(dets, meta)
    crop = meta.get("crop") or dominant_class(dets)
    per = ", ".join(f"{k[6:]} {v}" for k, v in feats.items() if k.startswith("count_")) or "-"
    row = {"file": file, "counts": per, "count": feats["count"], "total_area_px": round(feats["total_area_px"]),
           "weight_g": None, "low_g": None, "high_g": None, "measured_g": meta.get("total_weight_g"),
           "error_pct": None, "note": ""}
    try:
        est = model.predict(feats, crop)
    except NotImplementedError:
        row["note"] = todo_message(model.name)
        return row
    except (KeyError, ValueError, TypeError) as e:
        row["note"] = f"{model.name}: {e}"
        return row
    row.update(weight_g=round(est.weight_g, 4), note=est.note or "")
    if est.low_g is not None:
        row["low_g"] = round(est.low_g, 4)
    if est.high_g is not None:
        row["high_g"] = round(est.high_g, 4)
    m = row["measured_g"]
    if m:
        row["error_pct"] = round(100 * (est.weight_g - m) / m, 1)
    return row


def results_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def write_results_csv(rows: list[dict], out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(results_csv(rows), encoding="utf-8")
    return out


def overlay(path: Path, dets: list[dict], classes: list[str], out: Path, maxside: int = 1200) -> Path:
    with Image.open(path) as im:
        w = max(2, max(im.size) // 500)
        draw_boxes(im, [d["bbox"] for d in dets], out,
                   labels=[classes.index(d["label"]) if d["label"] in classes else 0 for d in dets],
                   scores=[d["score"] if d["score"] is not None else 0 for d in dets],
                   width=w, maxside=maxside, names=classes)
    return Path(out)


def run_inference(paths: list[Path], per_photo: dict[str, dict], project, *, ckpt: str, device: str,
                  score: float, tile: int, overlap: float, max_side: int, roi_text: str, model_name: str,
                  cache: Path, on_photo: Callable[[int, int, str], None] | None = None
                  ) -> tuple[list[dict], list[dict], list[str]]:
    """Run all photos. ``per_photo[file] = {crop, total_weight_g}`` (optional). ``on_photo(i, n, name)``
    is called before photo *i* (progress). Returns ``(rows, items, overlay_paths)``; *items* keep the
    detections for Save as samples."""
    cfg = W.load_config(project.workdir)
    model = load_weight_model(project.workdir, model_name, cfg)
    bundle = P.load_bundle(ckpt, device)
    rows, items, overlays = [], [], []
    for i, p in enumerate(paths):
        if on_photo:
            on_photo(i, len(paths), p.name)
        meta = {k: v for k, v in (per_photo.get(p.name) or {}).items() if v not in (None, "")}
        crop = meta.get("crop")
        try:
            dets = detect_photo(p, bundle, project, crop=crop, score=score, tile=tile, overlap=overlap,
                                max_side=max_side, roi_text=roi_text)
        except ValueError as e:                              # e.g. single-class model needs the crop
            rows.append({"file": p.name, "counts": "-", "count": None, "note": str(e)})
            continue
        if not meta.get("crop") and dets and len(project.classes) > 1:
            meta["crop"] = dominant_class(dets)
        row = estimate_row(p.name, dets, meta, model, cfg)
        rows.append(row)
        items.append({"path": str(p), "dets": dets, "meta": meta})
        overlays.append(str(overlay(p, dets, project.classes, Path(cache) / f"{p.stem}_pred.jpg")))
    return rows, items, overlays


# ---- save as samples ------------------------------------------------------------------------------
def save_samples(project, items: list[dict], proposer: str = "rfdetr") -> list[str]:
    """Copy photos into W/photos (never overwriting) and store the predicted boxes as origin=rfdetr,
    ``reviewed=False``, with crop / total_weight_g. Returns the stored file names."""
    store = project.store()
    names = []
    for it in items:
        name = L.import_photos([it["path"]], project.photos_dir)[0]
        with Image.open(project.photos_dir / name) as im:
            size = im.size
        meta = {k: it["meta"].get(k) for k in ("crop", "total_weight_g")
                if it["meta"].get(k) is not None}
        boxes = [{"bbox": d["bbox"], "class": d["label"], "origin": "rfdetr", "score": d["score"]}
                 for d in it["dets"] if d["label"] in project.classes]
        store.upsert_image(name, size[0], size[1], boxes, meta, reviewed=False, proposer=proposer)
        names.append(name)
    return names


def constants_rows(project) -> list[list]:
    cfg = W.load_config(project.workdir)
    cc, ac = cfg["constants"], cfg["area_constants"]
    crops = ["default", *[c for c in dict.fromkeys([*project.classes, *cc, *ac]) if c != "default"]]
    return [[c, cc.get(c), ac.get(c)] for c in crops]


def save_constants(project, rows) -> str:
    cfg = W.load_config(project.workdir)
    for c, g, a in rows:
        c = str(c).strip()
        if not c:
            continue
        try:
            if g not in (None, "") and g == g:
                cfg["constants"][c] = float(g)
            if a not in (None, "") and a == a:
                cfg["area_constants"][c] = float(a)
        except (TypeError, ValueError):
            return f"Could not read the numbers for '{c}'."
    W.save_config(project.workdir, cfg)
    return "Saved constants to weight_config.json."
