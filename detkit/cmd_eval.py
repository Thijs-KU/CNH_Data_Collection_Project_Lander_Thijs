"""``detkit eval``: evaluate predictions (or a checkpoint) against COCO ground truth. See docs/EVAL.md."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from . import coco as C
from .evaluate import evaluate, preds_from_json, write_report
from .project import Project


def _center_in(box, crop) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return crop[0] <= cx < crop[2] and crop[1] <= cy < crop[3]


def _run_model(a, gt_coco, images_dir: Path, level: str, manifest: dict | None, out: Path) -> tuple[dict, list[str]]:
    from PIL import Image

    from .predict import load_model, predict_image

    model, proc, meta, device = load_model(a.model)
    tp = meta.get("tile_params", {})
    overlap = a.overlap if a.overlap is not None else tp.get("overlap", 0.2)
    crop_a = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
    result = {}
    for img in gt_coco["images"]:
        f = img["file_name"]
        with Image.open(images_dir / f) as im:
            im = im.convert("RGB")
            if level == "tile":                       # the image IS one tile: single pass, no re-tiling
                tile, ov, crop = max(im.size), 0.0, None
            else:
                src = (manifest or {}).get("sources", {}).get(f, {})
                crop = crop_a or (tuple(src["crop"]) if src.get("crop") else None)
                tile, ov = a.tile or src.get("tile") or tp.get("tile") or 800, overlap
            dets = predict_image(im, model, proc, device, tile=tile, overlap=ov, crop=crop, score=a.score)
        result[f] = {"count": len(dets), "per_class": dict(Counter(d["label"] for d in dets)), "detections": dets}
        print(f"{f}: {len(dets)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result, list(model.config.id2label.values())


def cmd_eval(a) -> int:
    p = Project.load(a.workdir)
    if a.gt == "reviewed":
        gt_path, images_dir, level = p.reviewed_coco, p.photos_dir, "source"
    elif a.gt == "val":
        gt_path, images_dir, level = p.coco_dir / "val" / "labels.json", p.coco_dir / "val" / "images", "tile"
    else:
        gt_path, images_dir, level = Path(a.gt), Path(a.gt).parent / "images", a.level
    if a.images:
        images_dir = Path(a.images)
    gt = C.load(gt_path)
    manifest = C.load(p.manifest) if p.manifest.exists() else None
    if level in (None, "auto"):
        tiles = {t["file"] for t in (manifest or {}).get("tiles", [])}
        level = "tile" if tiles and {i["file_name"] for i in gt["images"]} <= tiles else "source"
    if not gt["images"]:
        print(f"no images in GT {gt_path}", file=sys.stderr)
        return 1

    name = a.name or (Path(a.model).parent.name if a.model else Path(a.pred).stem) + f"_{a.gt if a.gt in ('reviewed', 'val') else Path(a.gt).stem}"
    out = Path(a.out) if a.out else p.reports_dir / f"eval_{name}"
    model_labels = None
    if a.model:
        preds_json, model_labels = _run_model(a, gt, images_dir, level, manifest, out)
        pred_desc = f"model `{a.model}` run at score >= {a.score} ({level}-level inference)"
    else:
        preds_json = json.loads(Path(a.pred).read_text(encoding="utf-8"))
        pred_desc = f"file `{a.pred}`"
    preds = preds_from_json(preds_json)

    crops_note = ""
    if level == "source" and manifest:                # only score the region that was labelled/scanned
        crop_of = {f: (tuple(s["crop"]) if s.get("crop") else None) for f, s in manifest["sources"].items()}
        if a.crop:
            crop_of = {i["file_name"]: tuple(int(v) for v in a.crop.split(",")) for i in gt["images"]}
        if any(crop_of.get(i["file_name"]) for i in gt["images"]):
            keep = []
            names = {i["id"]: i["file_name"] for i in gt["images"]}
            for an in gt["annotations"]:
                x, y, w, h = an["bbox"]
                cr = crop_of.get(names[an["image_id"]])
                if cr is None or _center_in([x, y, x + w, y + h], cr):
                    keep.append(an)
            gt = {**gt, "annotations": keep}
            preds = {f: [d for d in v if crop_of.get(f) is None or _center_in(d[0], crop_of[f])]
                     for f, v in preds.items()}
            crops_note = " GT and predictions restricted to the scanned crop region (box centre inside)."

    if a.gt_note:
        note = a.gt_note
    elif a.gt == "reviewed":
        note = "human-reviewed annotations (reviewed_coco.json); still only as good as the review."
    else:
        note = ("this GT may be model pseudo-labels (e.g. the dataset's automatic labels); scores then measure "
                "agreement with those labels, NOT with the truth. Pass --gt-note to state its provenance.")
    res = evaluate(gt, preds, iou=a.iou, op_score=a.op_score, model_labels=model_labels, map_grid=a.map)
    meta = {"name": name, "gt": str(gt_path), "level": level, "pred_desc": pred_desc, "gt_note": note + crops_note}
    write_report(res, out, meta)
    b, mi, ca = res["best_f1"], res["micro"], res["count_area"]["op"]["count"]
    print(f"\nIoU {a.iou}  op score {a.op_score}: P={mi['precision']:.3f} R={mi['recall']:.3f} F1={mi['f1']:.3f} "
          f"AP={mi['ap']:.3f}  | best F1={b['f1']:.3f} @ score {b['score']}")
    print(f"count: bias={ca.get('bias', float('nan')):.2f} MAE={ca.get('mae', float('nan')):.2f} "
          f"MAPE={ca.get('mape_pct', float('nan')):.1f}% R2={ca.get('r2', float('nan')):.3f}")
    print(f"report: {out / 'eval.md'}")
    return 0


def register(sub) -> None:
    s = sub.add_parser("eval", help="P/R/F1, AP, PR/ROC, best threshold, count + area error vs COCO GT")
    s.add_argument("--workdir", default="runs/default")
    s.add_argument("--gt", required=True, help="reviewed | val | path to a COCO json")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred", help="predictions.json from `detkit predict`")
    g.add_argument("--model", help="checkpoint dir (…/final); runs inference (tiled for source-level GT)")
    s.add_argument("--score", type=float, default=0.05, help="score floor for --model inference")
    s.add_argument("--images", help="image dir for --model (default: photos/ for reviewed, coco/val/images for val)")
    s.add_argument("--iou", type=float, default=0.5)
    s.add_argument("--op-score", type=float, default=0.5, help="score threshold for the headline P/R/F1 and count error")
    s.add_argument("--map", action="store_true", help="also compute mAP over IoU 0.5:0.95")
    s.add_argument("--level", choices=["auto", "tile", "source"], default="auto", help="for --gt PATH")
    s.add_argument("--tile", type=int, help="--model on source-level GT: tile size (default: manifest/checkpoint)")
    s.add_argument("--overlap", type=float)
    s.add_argument("--crop", help="x0,y0,x1,y1 scan/score region (default: per-photo crop from tiles_manifest.json)")
    s.add_argument("--gt-note", help="provenance of the GT, printed in the report (e.g. 'automatic labels, unreviewed')")
    s.add_argument("--name")
    s.add_argument("--out", help="default: <workdir>/reports/eval_<name>")
    s.set_defaults(fn=cmd_eval)
