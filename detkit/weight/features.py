"""Per-photo features from detections, and the table of samples (photo + features + true weight)."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path


def _as_bgr(image):
    """Path, PIL image or BGR ndarray -> BGR uint8 ndarray (None if the file cannot be read)."""
    import cv2
    import numpy as np

    if isinstance(image, (str, Path)):
        return cv2.imread(str(image), cv2.IMREAD_COLOR)
    if hasattr(image, "convert"):                                   # PIL
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    return np.asarray(image)


def kernel_mask(image, bboxes: list, pad: int = 4):
    """Pixels that belong to a kernel, found inside the detection boxes only.

    ``image``: path / PIL / BGR array of the PREPROCESSED photo (the frame the boxes are in).
    ``bboxes``: ``[[x1, y1, x2, y2], ...]``. Returns ``(mask, areas)``: ``mask`` is a bool HxW array (union of
    all boxes, so overlapping boxes never count a pixel twice) and ``areas`` the kernel pixels per box.

    Method: the paper is the dominant colour, so every pixel gets its colour distance to the paper (median Lab
    colour outside the boxes; lightness counts half). Otsu's threshold on that distance, computed over the pixels in
    and around the boxes (so both clusters are always present), splits paper from kernel. Per box the mask is
    closed, reduced to its largest blob and filled to its convex hull (a kernel is convex; this recovers the pale
    tip that is almost paper-coloured).
    """
    import cv2
    import numpy as np

    bgr = _as_bgr(image)
    if bgr is None:
        raise ValueError(f"cannot read image {image!r}")
    H, W = bgr.shape[:2]
    mask = np.zeros((H, W), bool)
    boxes = []
    for x1, y1, x2, y2 in bboxes:
        b = (max(0, int(x1)), max(0, int(y1)), min(W, int(math.ceil(x2))), min(H, int(math.ceil(y2))))
        boxes.append(b if b[2] > b[0] and b[3] > b[1] else None)
    areas = [0] * len(boxes)
    live = [b for b in boxes if b]
    if not live:
        return mask, areas

    def covered(margin: int):
        r = np.zeros((H, W), bool)
        for x1, y1, x2, y2 in live:
            r[max(0, y1 - margin):y2 + margin, max(0, x1 - margin):x2 + margin] = True
        return r

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    outside = ~covered(6)
    paper = np.median(lab[outside] if outside.any() else lab.reshape(-1, 3), axis=0)
    d = lab - paper
    dist = cv2.GaussianBlur(np.sqrt(d[..., 1] ** 2 + d[..., 2] ** 2 + (0.5 * d[..., 0]) ** 2), (3, 3), 0)
    near = covered(pad)
    top = float(dist[near].max())
    if top <= 0:
        return mask, areas
    scaled = np.clip(dist * (255 / top), 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(scaled[near].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw = (scaled > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for i, b in enumerate(boxes):
        if not b:
            continue
        x1, y1, x2, y2 = b
        sub = cv2.morphologyEx(raw[y1:y2, x1:x2], cv2.MORPH_CLOSE, k)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(sub)
        if n < 2:
            continue
        comp = (lbl == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        comp = np.zeros_like(comp)
        cv2.fillPoly(comp, [cv2.convexHull(np.vstack(cnts))], 1)
        areas[i] = int(comp.sum())
        mask[y1:y2, x1:x2] |= comp.astype(bool)
    return mask, areas


def image_features(detections: list[dict], meta: dict | None = None, image=None) -> dict:
    """Flat feature dict for one photo.

    ``detections``: ``[{"bbox": [x1,y1,x2,y2], "score": .., "label": "wheat"}]`` in source pixels
    (the format of predictions.json; store boxes with ``"class"`` instead of ``"label"`` also work).
    Keys: count, count_<label>, total_area_px, mean_area_px, median_area_px, mean_diag_px and, when
    ``meta["scale_mm_per_px"]`` is set, total_area_mm2, mean_area_mm2. Area = box area (w*h), a proxy
    for the grain's true outline area (a box around a tilted grain over-estimates it).

    ``image`` (optional: path / PIL / BGR array of the PREPROCESSED photo the boxes belong to) adds the kernel
    features, which count only the pixels that really are kernel (see :func:`kernel_mask`), not the whole box:
    kernel_area_px (sum), mean_kernel_area_px, median_kernel_area_px, kernel_fill_ratio (kernel px / box px) and,
    with a scale, kernel_area_mm2 / mean_kernel_area_mm2.
    """
    areas, diags, labels, boxes = [], [], [], []
    for d in detections:
        x1, y1, x2, y2 = (float(v) for v in d["bbox"])
        boxes.append([x1, y1, x2, y2])
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        areas.append(w * h)
        diags.append(math.hypot(w, h))
        labels.append(d.get("label", d.get("class")))
    out: dict = {"count": len(areas)}
    for lab in sorted({lab for lab in labels if lab}):
        out[f"count_{lab}"] = labels.count(lab)
    out["total_area_px"] = sum(areas)
    out["mean_area_px"] = statistics.fmean(areas) if areas else 0.0
    out["median_area_px"] = statistics.median(areas) if areas else 0.0
    out["mean_diag_px"] = statistics.fmean(diags) if diags else 0.0
    scale = (meta or {}).get("scale_mm_per_px")
    if scale:
        out["total_area_mm2"] = out["total_area_px"] * scale ** 2
        out["mean_area_mm2"] = out["mean_area_px"] * scale ** 2
    if image is not None:
        mask, kareas = kernel_mask(image, boxes)
        out["kernel_area_px"] = float(mask.sum())
        out["mean_kernel_area_px"] = statistics.fmean(kareas) if kareas else 0.0
        out["median_kernel_area_px"] = statistics.median(kareas) if kareas else 0.0
        out["kernel_fill_ratio"] = out["kernel_area_px"] / out["total_area_px"] if out["total_area_px"] else 0.0
        if scale:
            out["kernel_area_mm2"] = out["kernel_area_px"] * scale ** 2
            out["mean_kernel_area_mm2"] = out["mean_kernel_area_px"] * scale ** 2
    return out


@dataclass
class Sample:
    """One photo: its features, the crop, and (if it was weighed) the true weight."""
    file: str
    crop: str | None
    features: dict
    weight_g: float | None = None
    meta: dict = field(default_factory=dict)

    def get(self, key: str):
        """Value of ``key`` from the sample itself (weight_g, crop) or its features."""
        return getattr(self, key) if key in ("weight_g", "crop", "file") else self.features.get(key)


class SampleTable:
    def __init__(self, rows: list[Sample] | None = None):
        self.rows: list[Sample] = list(rows or [])

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    # ---- building ------------------------------------------------------
    def add(self, file: str, detections: list[dict], meta: dict | None = None, image=None) -> Sample:
        meta = dict(meta or {})
        w = meta.get("total_weight_g")
        s = Sample(file, meta.get("crop"), image_features(detections, meta, image),
                   float(w) if w is not None else None, meta)
        self.rows.append(s)
        return s

    @classmethod
    def from_project(cls, project, *, reviewed_only: bool = False, use: str = "reviewed",
                     predictions_path: str | Path | None = None, kernel_area: bool = True) -> "SampleTable":
        """Boxes from the reviewed store (``use="reviewed"``) or a predictions.json (``use="predictions"``,
        default ``<workdir>/predictions/predictions.json``). Metadata (crop, weight, scale) always
        comes from the store. ``kernel_area``: also add the kernel-pixel features, measured on the stored
        (preprocessed) photo ``<workdir>/photos/<file>``; photos that are missing just get no kernel features."""
        store = project.store()
        meta = store.load_meta()
        t = cls()
        if use == "predictions":
            p = Path(predictions_path or project.workdir / "predictions" / "predictions.json")
            preds = {Path(k).name: v.get("detections", []) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
            files = list(preds)
            get = lambda f: preds[f]                                       # noqa: E731
        elif use == "reviewed":
            files = [i["file_name"] for i in store.load_coco()["images"]]
            get = lambda f: store.get_image(f)[0]                          # noqa: E731
        else:
            raise ValueError(f"use must be 'reviewed' or 'predictions', got {use!r}")
        for f in files:
            m = meta.get(f, {})
            if reviewed_only and not m.get("reviewed"):
                continue
            img = project.photos_dir / f
            t.add(f, get(f), m, img if kernel_area and img.exists() else None)
        return t

    # ---- views ---------------------------------------------------------
    def with_weight(self) -> "SampleTable":
        return SampleTable([s for s in self.rows if s.weight_g is not None])

    def inconsistent_piles(self, tol: float = 0.01) -> dict[str, list[Sample]]:
        """Piles (same ``pile_id``) whose weighed photos do not share one weight (they should: one pile, one
        balance reading), i.e. a typo in a ``total_weight_g``. ``{pile_id: [its weighed photos]}``; ``tol`` = relative
        difference that is still accepted."""
        by: dict[str, list[Sample]] = {}
        for s in self.with_weight().rows:
            pid = s.meta.get("pile_id")
            if pid not in (None, ""):
                by.setdefault(str(pid), []).append(s)
        return {k: v for k, v in by.items() if max(s.weight_g for s in v) > min(s.weight_g for s in v) * (1 + tol)}

    def for_crop(self, crop: str | None) -> "SampleTable":
        return SampleTable([s for s in self.rows if crop is None or s.crop == crop])

    def to_records(self) -> list[dict]:
        return [{"file": s.file, "crop": s.crop, "weight_g": s.weight_g, **s.features} for s in self.rows]

    def to_csv(self, path: str | Path) -> None:
        recs = self.to_records()
        keys = list(dict.fromkeys(k for r in recs for k in r))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(recs)
