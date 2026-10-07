"""Box proposals for the app: RF-DETR (tiled, local).

Boxes are returned in ORIGINAL photo pixels with ``origin="rfdetr"``. Heavy imports (torch,
transformers) happen lazily so the pure functions are testable without them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image

from . import logic as L


_MODELS: dict[tuple[str, str], tuple] = {}


# ---- checkpoints ---------------------------------------------------------------------------------
def _is_ckpt(d: Path) -> bool:
    return d.is_dir() and (d / "config.json").exists()


def find_checkpoints(workdir: Path, *preferred: str | None) -> list[str]:
    """Checkpoint dirs (contain config.json), preferred ones first, then ``W/checkpoints/*/final``,
    ``W/weights`` and ``./weights`` (each dir itself, its children and their ``final``)."""
    found: list[str] = []

    seen: set[Path] = set()

    def add(p) -> bool:
        if not (p and _is_ckpt(Path(p))):
            return False
        if Path(p).resolve() not in seen:
            seen.add(Path(p).resolve())
            found.append(str(p))
        return True

    root = Path(__file__).resolve().parents[2]              # the detkit checkout (holds weights/)
    for p in preferred:
        if p:
            add(p) or add(Path(workdir) / p) or add(root / p)
    for d in sorted((Path(workdir) / "checkpoints").glob("*/final")):
        add(d)
    for wroot in (Path(workdir) / "weights", Path("weights"), root / "weights"):
        add(wroot)
        if wroot.is_dir():
            for c in sorted(wroot.iterdir()):
                add(c / "final")
                add(c)
    return found


def load_bundle(ckpt: str, device: str = "auto"):
    """Cached ``(model, processor, meta, device)`` for a checkpoint dir."""
    from ..predict import load_model
    dev = None if device == "auto" else device
    key = (str(ckpt), device)
    if key not in _MODELS:
        _MODELS.clear()                                     # keep at most one model in memory
        _MODELS[key] = load_model(ckpt, dev)
    return _MODELS[key]


def norm_label(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", s.lower()).strip()


# ---- label mapping -------------------------------------------------------------------------------
def map_label(label: str, classes: list[str], crop: str | None, single_class_model: bool) -> str | None:
    """Project class for a detector label: by NAME (case/space-insensitive); a single-class model
    (e.g. the 'seed' baseline) maps everything to the photo's crop class; otherwise None (dropped)."""
    for c in classes:
        if norm_label(c) == norm_label(label):
            return c
    if single_class_model and crop in classes:
        return crop
    return None


def dets_to_boxes(dets: list[dict], classes: list[str], crop: str | None, *, origin: str,
                  single_class_model: bool = False, offset=(0, 0), factor: float = 1.0
                  ) -> tuple[list[dict], int]:
    """Detections (``bbox``, ``label``, ``score``) -> app boxes; undo ROI *offset* and the
    inference downscale *factor* (boxes / factor + offset). Returns ``(boxes, n_dropped)``."""
    out, dropped = [], 0
    for d in dets:
        cls = map_label(str(d.get("label", "")), classes, crop, single_class_model)
        if cls is None:
            dropped += 1
            continue
        x1, y1, x2, y2 = d["bbox"]
        out.append({"bbox": [x1 / factor + offset[0], y1 / factor + offset[1],
                             x2 / factor + offset[0], y2 / factor + offset[1]],
                    "class": cls, "origin": origin,
                    "score": round(float(d["score"]), 3) if d.get("score") is not None else None})
    return out, dropped


def parse_roi(text: str | None, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """``"x0,y0,x1,y1"`` (original px) -> clamped tuple, or None when empty."""
    if not text or not text.strip():
        return None
    try:
        x0, y0, x1, y1 = (int(float(v)) for v in text.replace(";", ",").split(","))
    except ValueError:
        raise ValueError("ROI must be 'x0,y0,x1,y1' in pixels of the original photo")
    W, H = size
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError(f"ROI {text!r} is empty for a {W}x{H} photo")
    return x0, y0, x1, y1


# ---- RF-DETR -----------------------------------------------------------------------------------
def propose_rfdetr(image: Image.Image, bundle, classes: list[str], crop: str | None, *,
                   score: float = 0.5, tile: int = 800, overlap: float = 0.2, max_side: int = 0,
                   roi: tuple[int, int, int, int] | None = None,
                   predict_fn: Callable | None = None) -> tuple[list[dict], dict]:
    """Tiled RF-DETR proposals for one photo. ``max_side`` > 0 downsizes the (ROI of the) photo so
    its longest side is at most that many px before tiling (CPU-friendly; boxes are scaled back)."""
    if predict_fn is None:
        from ..predict import predict_image as predict_fn
    model, proc, _meta, device = bundle
    id2label = getattr(getattr(model, "config", None), "id2label", {}) or {}
    single = len(id2label) <= 1
    x0, y0, x1, y1 = roi or (0, 0, *image.size)
    im = image.crop((x0, y0, x1, y1)) if roi else image
    factor = 1.0
    if max_side and max(im.size) > max_side:
        factor = max_side / max(im.size)
        im = im.resize((max(1, round(im.size[0] * factor)), max(1, round(im.size[1] * factor))),
                       Image.BILINEAR)
    dets = predict_fn(im.convert("RGB"), model, proc, device, tile=int(tile), overlap=float(overlap),
                      score=float(score))
    if single and crop not in classes and len(classes) > 1:
        raise ValueError("this checkpoint is single-class: choose the photo's crop first so the boxes "
                         "get the right class")
    boxes, dropped = dets_to_boxes(dets, classes, crop or (classes[0] if classes else None),
                                   origin="rfdetr", single_class_model=single,
                                   offset=(x0, y0), factor=factor)
    return boxes, {"detected": len(dets), "dropped_unknown_label": dropped, "factor": round(factor, 3),
                   "model_classes": list(id2label.values())}



def propose_photos(project, files: list[str], bundle, *, score: float = 0.5, tile: int = 800,
                   overlap: float = 0.2, max_side: int = 0, mode: str = L.MODES[0]
                   ) -> Iterator[tuple[str, int | None, str | None]]:
    """Batch proposals for photos in ``W/photos``: each photo's boxes are combined with the stored ones
    (*mode*, see ``logic.combine``) and saved UNREVIEWED; metadata is kept. Reviewed photos are never
    touched. A generator (one photo per step, so a caller can show progress or stop): yields
    ``(file, n_boxes, None)`` or ``(file, None, reason)`` when the photo was skipped."""
    store = project.store()
    for f in files:
        _old, meta = store.get_image(f)
        if meta.get("reviewed"):
            yield f, None, "already reviewed"
            continue
        crop = meta.get("crop") or (project.classes[0] if len(project.classes) == 1 else None)
        try:
            with Image.open(project.photos_dir / f) as im:
                size = im.size
                new, _info = propose_rfdetr(im.convert("RGB"), bundle, project.classes, crop, score=score,
                                            tile=tile, overlap=overlap, max_side=max_side)
        except (ValueError, OSError) as e:
            yield f, None, str(e)
            continue
        boxes = L.combine(store.get_image(f)[0], new, mode)
        store.upsert_image(f, size[0], size[1], boxes, {}, reviewed=False, proposer="rfdetr")
        yield f, len(boxes), None
