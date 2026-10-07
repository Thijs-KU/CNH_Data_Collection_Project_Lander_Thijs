"""Per-photo features from detections, and the table of samples (photo + features + true weight)."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path


def image_features(detections: list[dict], meta: dict | None = None) -> dict:
    """Flat feature dict for one photo.

    ``detections``: ``[{"bbox": [x1,y1,x2,y2], "score": .., "label": "wheat"}]`` in source pixels
    (the format of predictions.json; store boxes with ``"class"`` instead of ``"label"`` also work).
    Keys: count, count_<label>, total_area_px, mean_area_px, median_area_px, mean_diag_px and, when
    ``meta["scale_mm_per_px"]`` is set, total_area_mm2, mean_area_mm2. Area = box area (w*h), a proxy
    for the grain's true outline area (a box around a tilted grain over-estimates it).
    """
    areas, diags, labels = [], [], []
    for d in detections:
        x1, y1, x2, y2 = (float(v) for v in d["bbox"])
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
    def add(self, file: str, detections: list[dict], meta: dict | None = None) -> Sample:
        meta = dict(meta or {})
        w = meta.get("total_weight_g")
        s = Sample(file, meta.get("crop"), image_features(detections, meta),
                   float(w) if w is not None else None, meta)
        self.rows.append(s)
        return s

    @classmethod
    def from_project(cls, project, *, reviewed_only: bool = False, use: str = "reviewed",
                     predictions_path: str | Path | None = None) -> "SampleTable":
        """Boxes from the reviewed store (``use="reviewed"``) or a predictions.json (``use="predictions"``,
        default ``<workdir>/predictions/predictions.json``). Metadata (crop, weight, scale) always
        comes from the store."""
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
            t.add(f, get(f), m)
        return t

    # ---- views ---------------------------------------------------------
    def with_weight(self) -> "SampleTable":
        return SampleTable([s for s in self.rows if s.weight_g is not None])

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
