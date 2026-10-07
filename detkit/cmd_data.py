"""Data commands: ``import-photos`` and ``retile`` (reviewed source boxes -> tile-level COCO).

Also exports :func:`retile` / :func:`import_photos` for ``detkit split --from-reviewed`` (cli.py).
"""

from __future__ import annotations

import csv
import filecmp
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

from . import coco as C
from .project import Project
from .schema import IMAGE_FIELDS, coerce
from .tiling import IMAGE_EXT


# ---- import-photos ------------------------------------------------------------------------
def read_meta_file(path: str | Path, fields=IMAGE_FIELDS) -> dict[str, dict]:
    """``{file_name: {field: value}}`` from a CSV (column ``file`` + the *fields* keys) or JSON
    (``{file: {...}}`` or ``[{"file": ..., ...}]``). Unknown columns are ignored; empty -> absent."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = [{"file": k, **v} for k, v in raw.items()] if isinstance(raw, dict) else raw
    else:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    out: dict[str, dict] = {}
    for r in rows:
        name = str(r.get("file") or "").strip()
        if not name:
            continue
        out[Path(name).name] = {f.key: coerce(f, r[f.key]) for f in fields
                                if f.key in r and coerce(f, r[f.key]) is not None}
    return out


def _canon_crop(crop: str, classes: list[str]) -> str | None:
    return next((c for c in classes if c.lower() == crop.strip().lower()), None)


def import_photos(p: Project, images: str | Path, *, defaults: dict | None = None,
                  meta_rows: dict[str, dict] | None = None, force: bool = False,
                  add_class: str | None = None) -> dict:
    """Copy photos into ``W/photos`` and register empty, unreviewed Store entries."""
    src = Path(images)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXT)
    if not files:
        raise FileNotFoundError(f"no images in {src}")
    defaults = {k: v for k, v in (defaults or {}).items() if v is not None}
    meta_rows = meta_rows or {}
    if add_class and not _canon_crop(add_class, p.classes):
        p.classes.append(add_class.strip())
        new_class = add_class.strip()
    else:
        new_class = None
    # resolve + validate all metadata before touching the disk
    plan, problems = [], []
    for f in files:
        m = {**defaults, **meta_rows.get(f.name, {})}
        if m.get("crop") is not None:
            c = _canon_crop(str(m["crop"]), p.classes)
            if c is None:
                problems.append(f"{f.name}: crop '{m['crop']}' is not a project class {p.classes} "
                                "(use --add-class NAME to add it)")
            m["crop"] = c
        dst = p.photos_dir / f.name
        if dst.exists() and not filecmp.cmp(f, dst, shallow=False) and not force:
            problems.append(f"{f.name}: differs from existing {dst} (use --force to overwrite)")
        plan.append((f, dst, m))
    unknown_rows = sorted(set(meta_rows) - {f.name for f in files})
    if problems:
        raise ValueError("import aborted, nothing copied:\n  " + "\n  ".join(problems))

    if new_class:
        p.save()
    p.photos_dir.mkdir(parents=True, exist_ok=True)
    store = p.store()
    have = {i["file_name"] for i in store.load_coco()["images"]} | set(store.load_meta())
    res = {"copied": [], "identical_skipped": [], "overwritten": [], "registered": [],
           "already_in_store": [], "added_class": new_class, "unmatched_meta_rows": unknown_rows}
    for f, dst, m in plan:
        if dst.exists():
            if filecmp.cmp(f, dst, shallow=False):
                res["identical_skipped"].append(f.name)
            else:
                shutil.copy2(f, dst)
                res["overwritten"].append(f.name)
        else:
            shutil.copy2(f, dst)
            res["copied"].append(f.name)
        if f.name in have:
            res["already_in_store"].append(f.name)     # boxes/metadata untouched
            continue
        with Image.open(dst) as im:
            w, h = im.size
        store.upsert_image(f.name, w, h, [], m, reviewed=False)
        res["registered"].append(f.name)
    return res


def cmd_import(a) -> int:
    p = Project.load(a.workdir)
    defaults = {"crop": a.crop}
    rows = read_meta_file(a.meta, p.image_fields()) if a.meta else None
    try:
        r = import_photos(p, a.images, defaults=defaults, meta_rows=rows, force=a.force,
                          add_class=a.add_class)
    except (ValueError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 1
    print(f"copied {len(r['copied'])}, identical-skipped {len(r['identical_skipped'])}, "
          f"overwritten {len(r['overwritten'])}; registered {len(r['registered'])} new unreviewed "
          f"entries ({len(r['already_in_store'])} already in the store, left untouched) -> {p.photos_dir}")
    if r["added_class"]:
        print(f"added class '{r['added_class']}' -> classes={p.classes}")
    if r["unmatched_meta_rows"]:
        print(f"warning: metadata rows without a photo: {r['unmatched_meta_rows']}", file=sys.stderr)
    return 0


# ---- retile -------------------------------------------------------------------------------
def retile(p: Project, *, include_unreviewed: bool = False, min_visible: float = 0.6,
           min_px: float = 8.0, out: str | Path | None = None) -> dict:
    """Source-level reviewed boxes -> tile-level ``tiles_coco.json`` via ``tiles_manifest.json``.

    A box is clipped to each tile; kept when >= *min_visible* of its area lies inside the tile and
    the clipped box is >= *min_px* on both sides. Only tiles of selected (reviewed) photos are emitted,
    so unreviewed photos never enter training as false negatives. ``origin``/``score`` carry over.
    """
    if not p.manifest.exists():
        raise FileNotFoundError(f"{p.manifest} not found: run `detkit tile --workdir {p.workdir}` first")
    man = C.load(p.manifest)
    store = p.store()
    src = store.load_coco()
    meta = store.load_meta()
    names = C.class_names(src)
    by_id = {i["id"]: i["file_name"] for i in src["images"]}
    anns: dict[str, list[dict]] = {}
    for a in src["annotations"]:
        anns.setdefault(by_id[a["image_id"]], []).append(a)
    # A reviewed photo may legitimately have no boxes (negative example). An UNREVIEWED photo with no
    # boxes has unknown content (never proposed / labelling incomplete): never train on it.
    chosen = {i["file_name"] for i in src["images"]
              if meta.get(i["file_name"], {}).get("reviewed")
              or (include_unreviewed and anns.get(i["file_name"]))}

    tcoco = C.new_coco(names)
    stats = {"photos": 0, "photos_not_tiled": [],
             "photos_skipped_unlabelled": sorted(i["file_name"] for i in src["images"]
                                                 if i["file_name"] not in chosen),
             "boxes_in": 0, "boxes_kept": 0,
             "boxes_never_kept": 0, "tiles": 0}
    kept_src: dict[int, set] = {}
    for fname in sorted(chosen):
        if fname not in man["sources"]:
            stats["photos_not_tiled"].append(fname)
            continue
        stats["photos"] += 1
        boxes = anns.get(fname, [])
        stats["boxes_in"] += len(boxes)
        for t in (t for t in man["tiles"] if t["source"] == fname):
            img_id = C.add_image(tcoco, t["file"], t["w"], t["h"])
            stats["tiles"] += 1
            for a in boxes:
                x, y, w, h = a["bbox"]
                cx1, cy1 = max(x, t["x"]), max(y, t["y"])
                cx2, cy2 = min(x + w, t["x"] + t["w"]), min(y + h, t["y"] + t["h"])
                cw, ch = cx2 - cx1, cy2 - cy1
                if cw <= 0 or ch <= 0 or w <= 0 or h <= 0:
                    continue
                if cw * ch / (w * h) < min_visible or cw < min_px or ch < min_px:
                    continue
                extra = {k: a[k] for k in ("origin", "score") if a.get(k) is not None}
                C.add_box(tcoco, img_id, a["category_id"],
                          [cx1 - t["x"], cy1 - t["y"], cx2 - t["x"], cy2 - t["y"]], **extra)
                stats["boxes_kept"] += 1
                kept_src.setdefault(a["id"], set()).add(fname)
        stats["boxes_never_kept"] += sum(1 for a in boxes if a["id"] not in kept_src)
    C.save(tcoco, out or p.tiles_coco)
    stats["per_class"] = C.count_per_class(tcoco)
    stats["note"] = ("boxes_kept counts tile copies (a box in an overlap appears in several tiles); "
                     "boxes_never_kept = source boxes that fit no tile well enough")
    return stats


def cmd_retile(a) -> int:
    p = Project.load(a.workdir)
    try:
        s = retile(p, include_unreviewed=a.include_unreviewed, min_visible=a.min_visible,
                   min_px=a.min_px)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    print(json.dumps(s, indent=2))
    if not s["photos"]:
        print("no reviewed photos to retile (review in the app, or pass --include-unreviewed)",
              file=sys.stderr)
        return 1
    print(f"-> {p.tiles_coco}")
    return 0


def register(sub) -> None:
    s = sub.add_parser("import-photos", help="copy photos into <workdir>/photos + create unreviewed entries")
    s.add_argument("--workdir", default="runs/default")
    s.add_argument("--images", required=True, help="directory of photos to import")
    s.add_argument("--crop", help="crop/class of all these photos (a project class)")
    s.add_argument("--meta", help="CSV/JSON with a `file` column + metadata columns (schema.IMAGE_KEYS + extra properties)")
    s.add_argument("--add-class", help="append this class to the project if it is new")
    s.add_argument("--force", action="store_true", help="overwrite differing photos of the same name")
    s.set_defaults(fn=cmd_import)

    s = sub.add_parser("retile", help="reviewed source boxes -> tile-level annotations/tiles_coco.json")
    s.add_argument("--workdir", default="runs/default")
    s.add_argument("--include-unreviewed", action="store_true")
    s.add_argument("--min-visible", type=float, default=0.6,
                   help="min fraction of the box area inside a tile to keep it there")
    s.add_argument("--min-px", type=float, default=8.0, help="min clipped box side (px)")
    s.set_defaults(fn=cmd_retile)
