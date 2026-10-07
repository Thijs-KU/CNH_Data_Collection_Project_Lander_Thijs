"""Tiled inference on full-size photos: tile -> detect -> shift back -> per-class NMS -> draw + count."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image

from .merge import nms_per_class
from .tiling import IMAGE_EXT, plan_tiles
from .train import META
from .viz import draw_boxes


def load_model(ckpt: str | Path, device: str | None = None):
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForObjectDetection.from_pretrained(str(ckpt)).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(str(ckpt))
    meta_p = Path(ckpt) / META
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return model, proc, meta, device


def predict_image(image: Image.Image, model, proc, device: str, *, tile: int, overlap: float,
                  crop=None, score: float = 0.5, batch: int = 4,
                  iou_thr: float = 0.4, ioma_thr: float = 0.7) -> list[dict]:
    import torch

    rects = plan_tiles(image.size, crop, tile, overlap)
    boxes, labels, scores = [], [], []
    for i in range(0, len(rects), batch):
        chunk = rects[i:i + batch]
        crops = [image.crop((x, y, x + w, y + h)) for x, y, w, h in chunk]
        inputs = proc(images=crops, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        res = proc.post_process_object_detection(
            out, threshold=score, target_sizes=[c.size[::-1] for c in crops])
        for (x, y, _, _), r in zip(chunk, res):
            for b, s, l in zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist()):
                boxes.append([b[0] + x, b[1] + y, b[2] + x, b[3] + y])
                scores.append(s)
                labels.append(int(l))
    keep = nms_per_class(boxes, labels, scores, iou_thr=iou_thr, ioma_thr=ioma_thr)
    id2label = model.config.id2label
    return [{"bbox": [round(v, 1) for v in boxes[k]], "score": round(scores[k], 4),
             "label": id2label.get(labels[k], str(labels[k])), "label_id": labels[k]} for k in keep]


def predict_paths(ckpt: str | Path, inputs: list[str | Path], out_dir: str | Path, *,
                  tile: int | None = None, overlap: float | None = None, crop=None,
                  score: float = 0.5, device: str | None = None, batch: int = 4,
                  threads: int | None = None, max_side: int | None = None,
                  overlay: bool = True) -> dict:
    """``max_side`` downscales the photo before tiling (boxes are scaled back; ``tile``/``crop`` are
    then in downscaled pixels - use it for speed on CPU). ``threads`` sets torch CPU threads."""
    import torch

    if threads:
        torch.set_num_threads(threads)
    model, proc, meta, device = load_model(ckpt, device)
    tp = meta.get("tile_params", {})
    tile = tile or tp.get("tile") or 800
    overlap = tp.get("overlap", 0.2) if overlap is None else overlap
    files: list[Path] = []
    for p in map(Path, inputs):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT) if p.is_dir() else [p]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(model.config.id2label.values())
    result: dict = {}
    for f in files:
        with Image.open(f) as im:
            im = im.convert("RGB")
            orig_size = im.size
            work = im
            if max_side and max(im.size) > max_side:
                k = max_side / max(im.size)
                work = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
            dets = predict_image(work, model, proc, device, tile=tile, overlap=overlap,
                                 crop=crop, score=score, batch=batch)
            if work is not im:                      # back to original pixel coordinates
                sx, sy = orig_size[0] / work.width, orig_size[1] / work.height
                for d in dets:
                    x1, y1, x2, y2 = d["bbox"]
                    d["bbox"] = [round(x1 * sx, 1), round(y1 * sy, 1), round(x2 * sx, 1), round(y2 * sy, 1)]
            if overlay:
                draw_boxes(im, [d["bbox"] for d in dets], out_dir / f"{f.stem}_pred.jpg",
                           labels=[d["label_id"] for d in dets], scores=[d["score"] for d in dets],
                           width=max(3, round(max(im.size) / 250)), maxside=1600, names=names)
        result[f.name] = {"count": len(dets), "per_class": dict(Counter(d["label"] for d in dets)),
                          "detections": dets}
        print(f"{f.name}: {len(dets)} {dict(Counter(d['label'] for d in dets))}", flush=True)
    (out_dir / "predictions.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
