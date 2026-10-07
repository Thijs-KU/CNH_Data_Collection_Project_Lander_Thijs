"""Store for human-reviewed annotations + metadata (the app writes it; export/eval/train read it).

Layout inside a workdir::

    photos/                      copies of the source photos (full resolution)
    annotations/reviewed_coco.json   COCO on the SOURCE photos (full-res pixel coords)
    meta/image_meta.json         {file_name: {schema.IMAGE_FIELDS..., extra properties, reviewed, reviewed_at, proposer}}

``reviewed_coco.json`` annotation extras: ``origin`` (rfdetr|manual), ``score``, and the
per-box fields of ``schema.BOX_FIELDS`` (``weight_g``, ``box_notes``). Categories = project classes,
ids 0..K-1 in project order; classes may be appended later (never reordered).
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from . import coco as C
from .schema import BOX_KEYS, IMAGE_KEYS


class Store:
    def __init__(self, workdir: str | Path, classes: list[str], extra_image_keys: list[str] = ()):
        self.workdir = Path(workdir)
        self.classes = list(classes)
        self.image_keys = [*IMAGE_KEYS, *extra_image_keys]      # metadata keys upsert_image accepts
        self.photos_dir = self.workdir / "photos"
        self.coco_path = self.workdir / "annotations" / "reviewed_coco.json"
        self.meta_path = self.workdir / "meta" / "image_meta.json"

    # ---- io -----------------------------------------------------------
    def load_coco(self) -> dict:
        if self.coco_path.exists():
            coco = C.load(self.coco_path)
            have = C.class_names(coco)
            for c in self.classes[len(have):]:          # classes appended since last save
                coco["categories"].append({"id": len(coco["categories"]), "name": c})
            return coco
        return C.new_coco(self.classes)

    def load_meta(self) -> dict[str, dict]:
        return json.loads(self.meta_path.read_text(encoding="utf-8")) if self.meta_path.exists() else {}

    def save(self, coco: dict, meta: dict[str, dict]) -> None:
        C.save(coco, self.coco_path)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    # ---- high level ---------------------------------------------------
    def upsert_image(self, file_name: str, width: int, height: int, boxes: list[dict],
                     image_meta: dict | None = None, *, reviewed: bool = True,
                     proposer: str | None = None) -> None:
        """Replace all boxes + merge metadata for one photo.

        ``boxes``: ``[{"bbox": [x1,y1,x2,y2], "class": "wheat" | "category_id": 0,
        "origin": "manual", "score": 0.9, "weight_g": None, "box_notes": ""}]`` (source pixels).
        """
        coco, meta = self.load_coco(), self.load_meta()
        cid = {c: i for i, c in enumerate(C.class_names(coco))}
        img = next((i for i in coco["images"] if i["file_name"] == file_name), None)
        if img is None:
            img_id = C.add_image(coco, file_name, width, height)
        else:
            img_id = img["id"]
            coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] != img_id]
        for b in boxes:
            k = b["category_id"] if "category_id" in b else cid[b["class"]]
            extra = {kk: b.get(kk) for kk in ("origin", "score", *BOX_KEYS) if b.get(kk) is not None}
            C.add_box(coco, img_id, k, b["bbox"], **extra)
        for i, a in enumerate(coco["annotations"]):      # keep ids dense/unique after deletions
            a["id"] = i
        m = meta.get(file_name, {})
        m.update({k: v for k, v in (image_meta or {}).items() if k in self.image_keys})
        m["reviewed"] = bool(reviewed)
        if reviewed:
            m["reviewed_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        if proposer:
            m["proposer"] = proposer
        meta[file_name] = m
        self.save(coco, meta)

    def get_image(self, file_name: str) -> tuple[list[dict], dict]:
        """``(boxes, meta)`` for a photo; boxes carry ``class`` names and source-pixel ``bbox``."""
        coco, meta = self.load_coco(), self.load_meta()
        names = C.class_names(coco)
        img = next((i for i in coco["images"] if i["file_name"] == file_name), None)
        boxes = []
        if img:
            for a in coco["annotations"]:
                if a["image_id"] == img["id"]:
                    x, y, w, h = a["bbox"]
                    boxes.append({"bbox": [x, y, x + w, y + h], "class": names[a["category_id"]],
                                  **{k: a.get(k) for k in ("origin", "score", *BOX_KEYS)}})
        return boxes, meta.get(file_name, {})

    def reviewed_files(self) -> list[str]:
        return sorted(f for f, m in self.load_meta().items() if m.get("reviewed"))
