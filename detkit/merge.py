"""Map tile-level boxes back to the source photo and de-duplicate across tile overlaps."""

from __future__ import annotations

from . import coco as C


def _inter(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _area(a) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def iou(a, b) -> float:
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u else 0.0


def ioma(a, b) -> float:
    """Intersection over the smaller box (catches a small box nested in a larger one)."""
    return _inter(a, b) / max(1e-9, min(_area(a), _area(b)))


def nms(boxes: list, scores: list[float] | None = None, iou_thr: float = 0.4,
        ioma_thr: float = 0.7) -> list[int]:
    """Greedy suppression; returns kept indices. Ranks by score if given, else by area."""
    key = (lambda i: -scores[i]) if scores is not None else (lambda i: -_area(boxes[i]))
    keep: list[int] = []
    for i in sorted(range(len(boxes)), key=key):
        if all(iou(boxes[i], boxes[k]) < iou_thr and ioma(boxes[i], boxes[k]) < ioma_thr
               for k in keep):
            keep.append(i)
    return keep


def nms_per_class(boxes: list, labels: list[int], scores: list[float] | None = None, **kw) -> list[int]:
    keep: list[int] = []
    for c in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == c]
        sub = nms([boxes[i] for i in idx], [scores[i] for i in idx] if scores else None, **kw)
        keep.extend(idx[j] for j in sub)
    return sorted(keep)


def tiles_to_source(tiles_coco: dict, manifest: dict, iou_thr: float = 0.4,
                    ioma_thr: float = 0.7) -> dict:
    """Shift tile boxes by the tile offset, then per-class NMS per source photo."""
    by_tile = {t["file"]: t for t in manifest["tiles"]}
    per_src: dict[str, tuple[list, list]] = {}
    for fname, items in C.boxes_by_file(tiles_coco).items():
        t = by_tile[fname]
        boxes, labels = per_src.setdefault(t["source"], ([], []))
        for (x1, y1, x2, y2), cat in items:
            boxes.append([x1 + t["x"], y1 + t["y"], x2 + t["x"], y2 + t["y"]])
            labels.append(cat)
    out = C.new_coco(C.class_names(tiles_coco))
    for name, meta in sorted(manifest["sources"].items()):
        img_id = C.add_image(out, name, meta["width"], meta["height"])
        boxes, labels = per_src.get(name, ([], []))
        for i in nms_per_class(boxes, labels, iou_thr=iou_thr, ioma_thr=ioma_thr):
            C.add_box(out, img_id, labels[i], boxes[i])
    return out
