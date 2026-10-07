"""Build the train/val COCO folder the trainer expects, split by *source photo*.

Tiles of one photo (and near-duplicate burst shots) must never straddle train/val or the
validation mAP is inflated. Pass ``val_sources`` explicitly (recommended when you took bursts)
or let ``val_frac`` pick sources at random (seeded).
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from . import coco as C


def build_dataset(tiles_coco: dict, manifest: dict, tiles_dir: str | Path, out_dir: str | Path, *,
                  val_sources: list[str] | None = None, val_frac: float = 0.2, seed: int = 0) -> dict:
    tiles_dir, out_dir = Path(tiles_dir), Path(out_dir)
    src_of = {t["file"]: t["source"] for t in manifest["tiles"]}
    by_src: dict[str, list[dict]] = defaultdict(list)
    for im in tiles_coco["images"]:
        by_src[src_of[im["file_name"]]].append(im)
    sources = sorted(by_src)

    if val_sources is None:
        n_val = round(len(sources) * val_frac) if len(sources) > 1 else 0
        val_sources = sorted(random.Random(seed).sample(sources, n_val)) if n_val else []
    unknown = set(val_sources) - set(sources)
    if unknown:
        raise ValueError(f"val sources not among labelled sources: {sorted(unknown)}")
    val_set = set(val_sources)

    anns_by_img = defaultdict(list)
    for a in tiles_coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    classes = C.class_names(tiles_coco)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out = {s: C.new_coco(classes) for s in ("train", "val")}
    info: dict = {"train": {}, "val": {}}
    for src in sources:
        s = "val" if src in val_set else "train"
        for im in by_src[src]:
            dst = out_dir / s / "images"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(tiles_dir / im["file_name"], dst / im["file_name"])
            new_id = C.add_image(out[s], im["file_name"], im["width"], im["height"])
            for a in anns_by_img[im["id"]]:
                x, y, w, h = a["bbox"]
                C.add_box(out[s], new_id, a["category_id"], [x, y, x + w, y + h])
            e = info[s].setdefault(src, {"tiles": 0, "boxes": 0})
            e["tiles"] += 1
            e["boxes"] += len(anns_by_img[im["id"]])
    for s in ("train", "val"):
        if out[s]["images"]:
            C.save(out[s], out_dir / s / "labels.json", indent=None)
    report = {
        "split_by": "source photo", "val_sources": sorted(val_set), "classes": classes,
        "per_source": info,
        "totals": {s: {"tiles": len(out[s]["images"]), "boxes": len(out[s]["annotations"]),
                       "per_class": C.count_per_class(out[s])} for s in out},
        "warnings": [],
    }
    for c in classes:
        for s in ("train", "val"):
            if out[s]["images"] and report["totals"][s]["per_class"][c] == 0:
                report["warnings"].append(f"class '{c}' has 0 boxes in {s}")
    if not out["train"]["images"]:
        raise ValueError("no training tiles (all sources were assigned to val?)")
    (out_dir / "split.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
