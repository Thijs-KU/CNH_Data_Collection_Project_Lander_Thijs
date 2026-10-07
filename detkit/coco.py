"""Tiny COCO helpers (boxes are COCO ``[x, y, w, h]`` in files, ``[x1, y1, x2, y2]`` in code)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def new_coco(classes: list[str]) -> dict:
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": c} for i, c in enumerate(classes)],
    }


def add_image(coco: dict, file_name: str, width: int, height: int) -> int:
    img_id = len(coco["images"])
    coco["images"].append({"id": img_id, "file_name": file_name, "width": width, "height": height})
    return img_id


def add_box(coco: dict, image_id: int, category_id: int, xyxy, **extra) -> None:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return
    coco["annotations"].append({
        "id": len(coco["annotations"]), "image_id": image_id, "category_id": category_id,
        "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
        "area": round(w * h, 1), "iscrowd": 0, **extra,
    })


def boxes_by_file(coco: dict) -> dict[str, list[tuple[list[float], int]]]:
    """``{file_name: [([x1,y1,x2,y2], category_id), ...]}`` (images with no boxes map to [])."""
    names = {i["id"]: i["file_name"] for i in coco["images"]}
    out: dict[str, list] = {n: [] for n in names.values()}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        out[names[a["image_id"]]].append(([x, y, x + w, y + h], a["category_id"]))
    return out


def class_names(coco: dict) -> list[str]:
    return [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]


def count_per_class(coco: dict) -> dict[str, int]:
    names = {c["id"]: c["name"] for c in coco["categories"]}
    cnt: dict[str, int] = defaultdict(int)
    for a in coco["annotations"]:
        cnt[names[a["category_id"]]] += 1
    return {n: cnt.get(n, 0) for n in names.values()}


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(coco: dict, path: str | Path, indent: int | None = 1) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(coco, indent=indent), encoding="utf-8")
