"""Pure (Gradio-free) logic of the annotation app: form <-> metadata, box conversions, stats.

Conventions: boxes are dicts ``{"bbox": [x1,y1,x2,y2] (ORIGINAL pixels), "class": str,
"origin": rfdetr|manual, "score": float|None, "weight_g": float|None, "box_notes": str|None}``.
The browser shows a downscaled *preview* of large photos; ``scale = original / preview`` converts
back, so stored boxes are always in the original pixel grid.
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

from PIL import Image

from .. import coco as C
from ..merge import iou
from ..schema import AUTO_IMAGE_KEYS, BOX_FIELDS, EXTRA_KINDS, IMAGE_FIELDS, Field, coerce
from ..tiling import IMAGE_EXT
from ..viz import PALETTE

PREVIEW_MAXSIDE = 1600
BOX_KEYS = [f.key for f in BOX_FIELDS]
EXTRA_KEYS = ("origin", "score", *BOX_KEYS)


# ---- colours / labels ------------------------------------------------------------------------
def class_colors(classes: list[str]) -> list[str]:
    return ["#%02x%02x%02x" % PALETTE[i % len(PALETTE)] for i in range(len(classes))]


def field_label(f: Field) -> str:
    return f"{f.key} ({f.unit})" if f.unit else f.key


# ---- form <-> metadata -----------------------------------------------------------------------
def meta_to_form(meta: dict, fields: list[Field] = IMAGE_FIELDS) -> list:
    """Values for the generated form components, in *fields* order (None = empty)."""
    return [meta.get(f.key) for f in fields]


def form_to_meta(values: list, prior: dict | None = None,
                 fields: list[Field] = IMAGE_FIELDS) -> tuple[dict, list[str]]:
    """Coerce form values to typed metadata. Returns ``(meta, errors)``.

    Empty fields are omitted, except that a field that had a value in *prior* is set to None
    (Store merges keys, so that is how a field is cleared).
    """
    prior = prior or {}
    meta: dict = {}
    errors: list[str] = []
    for f, v in zip(fields, values):
        try:
            c = coerce(f, v)
        except (TypeError, ValueError):
            errors.append(f"{f.key}: cannot read {v!r} as {f.kind}")
            continue
        if c is None:
            if prior.get(f.key) is not None:
                meta[f.key] = None
        else:
            meta[f.key] = c
    return meta, errors


# ---- preview / geometry -----------------------------------------------------------------------
def make_preview(src: Path, cache_dir: Path, maxside: int = PREVIEW_MAXSIDE
                 ) -> tuple[Path, tuple[int, int], float]:
    """JPEG preview (longest side <= *maxside*). Returns ``(path, (W, H) original, scale)`` with
    ``scale = original_width / preview_width`` (1.0 when no downscale). Pixels are NOT
    EXIF-rotated, matching how the rest of detkit reads photos."""
    src = Path(src)
    with Image.open(src) as im:
        W, H = im.size
        if max(W, H) <= maxside and src.suffix.lower() in (".jpg", ".jpeg", ".png"):
            return src, (W, H), 1.0
        st = src.stat()
        out = Path(cache_dir) / f"{src.stem}_{st.st_size}_{int(st.st_mtime)}_{maxside}.jpg"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            p = im.convert("RGB")
            p.thumbnail((maxside, maxside))
            p.save(out, quality=90)
            pw = p.size[0]
        else:
            with Image.open(out) as q:
                pw = q.size[0]
    return out, (W, H), W / pw


def clamp_box(b, size) -> list[float] | None:
    W, H = size
    x1, y1, x2, y2 = (float(v) for v in b)
    x1, x2 = sorted((min(max(x1, 0), W), min(max(x2, 0), W)))
    y1, y2 = sorted((min(max(y1, 0), H), min(max(y2, 0), H)))
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)] if x2 - x1 >= 1 and y2 - y1 >= 1 else None


def box_area(b: dict) -> float:
    x1, y1, x2, y2 = b["bbox"]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def to_annotator(boxes: list[dict], scale: float, classes: list[str]) -> list[dict]:
    """Original-pixel boxes -> ``gradio_image_annotation`` boxes (preview pixels + class colour)."""
    cols = class_colors(classes)
    out = []
    for b in boxes:
        x1, y1, x2, y2 = (v / scale for v in b["bbox"])
        k = classes.index(b["class"]) if b["class"] in classes else 0
        c = cols[k].lstrip("#")
        out.append({"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2, "label": b["class"],
                    "color": tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))})
    return out


def from_annotator(ann_boxes: list[dict] | None, prev: list[dict], scale: float,
                   size: tuple[int, int], classes: list[str], default_class: str | None = None
                   ) -> list[dict]:
    """Editor boxes -> stored boxes, carrying per-box extras over from *prev*.

    Matching is by geometry (the component keeps no ids): a box within ~1 preview px of a previous
    one is *unchanged* (exact old coords + extras kept); otherwise the best-overlapping unmatched
    previous box (IoU >= 0.3) is treated as *moved/resized* (extras kept, origin -> manual);
    anything else is a *new manual* box. The class comes from the editor label.
    """
    fallback = default_class if default_class in classes else (classes[0] if classes else "")
    new = []
    for a in ann_boxes or []:
        bb = clamp_box([a["xmin"] * scale, a["ymin"] * scale, a["xmax"] * scale, a["ymax"] * scale], size)
        if bb is None:
            continue
        label = str(a.get("label") or "")
        new.append((bb, label if label in classes else fallback))
    tol = max(1.0, scale)
    used: set[int] = set()
    match: dict[int, int] = {}
    for i, (bb, _) in enumerate(new):                       # pass 1: unchanged geometry
        for j, p in enumerate(prev):
            if j not in used and all(abs(u - v) <= tol for u, v in zip(bb, p["bbox"])):
                match[i] = j
                used.add(j)
                break
    pairs = sorted(((iou(new[i][0], p["bbox"]), i, j) for i in range(len(new)) if i not in match
                    for j, p in enumerate(prev) if j not in used), reverse=True)
    moved: set[int] = set()
    for s, i, j in pairs:                                   # pass 2: moved / resized
        if s < 0.3:
            break
        if i not in match and j not in used:
            match[i] = j
            used.add(j)
            moved.add(i)
    out = []
    for i, (bb, cls) in enumerate(new):
        if i in match:
            p = prev[match[i]]
            b = {**p, "class": cls}
            if i in moved:
                b.update(bbox=bb, origin="manual", score=None)
        else:
            b = {"bbox": bb, "class": cls, "origin": "manual", "score": None}
        out.append(b)
    return out


# ---- per-box table -----------------------------------------------------------------------------
def table_headers() -> list[str]:
    return ["#", "class", "origin", "score", "area_px", *BOX_KEYS]


def boxes_to_rows(boxes: list[dict]) -> list[list]:
    return [[i + 1, b["class"], b.get("origin") or "", b.get("score"), round(box_area(b)),
             *[b.get(k) for k in BOX_KEYS]] for i, b in enumerate(boxes)]


def delete_boxes(boxes: list[dict], numbers: str) -> tuple[list[dict], list[int]]:
    """Remove boxes by their table number (1-based); *numbers* like ``"3"`` or ``"2, 5"``.
    Returns the remaining boxes and the numbers removed; raises ValueError on bad input."""
    try:
        nums = sorted({int(float(t)) for t in str(numbers or "").replace(";", ",").split(",") if t.strip()})
    except ValueError:
        raise ValueError(f"box number(s) must be integers, got {numbers!r}")
    if not nums:
        raise ValueError("enter the # of the box to delete (click its row in the box table)")
    bad = [n for n in nums if not 1 <= n <= len(boxes)]
    if bad:
        raise ValueError(f"no box {', '.join(map(str, bad))} (this photo has {len(boxes)} boxes)")
    return [b for i, b in enumerate(boxes, 1) if i not in nums], nums


def apply_table_edits(boxes: list[dict], rows) -> tuple[list[dict], list[str]]:
    """Copy edited BOX_FIELDS cells (weight_g, box_notes, ...) from the table back to *boxes*.
    Rows are matched by position; other columns are read-only."""
    errors: list[str] = []
    rows = [list(r) for r in rows]
    if len(rows) != len(boxes):
        return boxes, errors
    off = len(table_headers()) - len(BOX_KEYS)
    out = []
    for i, (b, r) in enumerate(zip(boxes, rows)):
        b = dict(b)
        for f, v in zip(BOX_FIELDS, r[off:]):
            if isinstance(v, float) and math.isnan(v):
                v = None
            try:
                b[f.key] = coerce(f, v)
            except (TypeError, ValueError):
                errors.append(f"box {i + 1} {f.key}: cannot read {v!r}")
        out.append(b)
    return out, errors


# ---- derived stats -----------------------------------------------------------------------------
def compute_stats(boxes: list[dict], meta: dict, classes: list[str]) -> dict:
    n = len(boxes)
    areas = [box_area(b) for b in boxes]
    s = meta.get("scale_mm_per_px")
    per_class = {c: sum(1 for b in boxes if b["class"] == c) for c in classes}
    st = {"n": n, "per_class": per_class, "total_area_px": sum(areas),
          "mean_area_px": (sum(areas) / n) if n else None}
    if s:
        st["total_area_mm2"] = st["total_area_px"] * s * s
        st["mean_area_mm2"] = st["mean_area_px"] * s * s if n else None
    w = [b.get("weight_g") for b in boxes if b.get("weight_g") is not None]
    st["boxes_weighed"], st["sum_box_weight_g"] = len(w), (sum(w) if w else None)
    tw = meta.get("total_weight_g")
    st["weight_hint"] = weight_hint(n, tw, len(w), st["sum_box_weight_g"], meta.get("manual_count"))
    return st


def weight_hint(n: int, total: float | None, n_weighed: int, sum_w: float | None,
                manual_count: int | None = None) -> str:
    parts = []
    if manual_count is not None and n and manual_count != n:
        parts.append(f"manual count {manual_count} differs from {n} boxes - missing or extra boxes?")
    if total is None:
        parts.append("Enter total_weight_g to get a consistency check." if n_weighed else "")
    elif n_weighed == 0:
        parts.append(f"total {total:g} g over {n} boxes = {total / n * 1000:.1f} mg per grain on average."
                     if n else f"total {total:g} g but no boxes yet.")
    else:
        diff = sum_w - total
        pct = 100 * diff / total if total else float("inf")
        note = "OK" if abs(pct) <= 5 else ("boxes weigh MORE than the total" if diff > 0
                                          else "some grains not weighed / missing")
        parts.append(f"{n_weighed}/{n} boxes weighed: sum {sum_w:g} g vs total {total:g} g "
                     f"({diff:+.4g} g, {pct:+.1f}%) - {note}.")
    return " ".join(p for p in parts if p)


def stats_markdown(st: dict, meta: dict) -> str:
    pc = ", ".join(f"**{c}** {k}" for c, k in st["per_class"].items() if k) or "no boxes"
    lines = [f"**{st['n']} boxes** - {pc}"]
    if st["n"]:
        lines.append(f"Box area: total {st['total_area_px']:,.0f} px, mean {st['mean_area_px']:,.0f} px")
        if "total_area_mm2" in st:
            lines[-1] += f" = total {st['total_area_mm2']:,.1f} mm2, mean {st['mean_area_mm2']:,.2f} mm2"
    if st["weight_hint"]:
        lines.append(f"Weight check: {st['weight_hint']}")
    return "  \n".join(lines)


# ---- photos / classes ----------------------------------------------------------------------------
def list_photos(photos_dir: Path) -> list[str]:
    d = Path(photos_dir)
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in IMAGE_EXT) if d.is_dir() else []


def import_photos(paths: list[str | Path], photos_dir: Path) -> list[str]:
    """Copy files into *photos_dir* (never overwriting: ``name_1.jpg`` ...). Returns new names."""
    photos_dir = Path(photos_dir)
    photos_dir.mkdir(parents=True, exist_ok=True)
    added = []
    for p in map(Path, paths):
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        dest, k = photos_dir / p.name, 1
        while dest.exists():
            dest = photos_dir / f"{p.stem}_{k}{p.suffix}"
            k += 1
        shutil.copy(p, dest)
        added.append(dest.name)
    return added


def photo_rows(files: list[str], store) -> list[list]:
    meta = store.load_meta()
    n_boxes = C.boxes_by_file(store.load_coco()) if store.coco_path.exists() else {}
    return [[f, "reviewed" if meta.get(f, {}).get("reviewed") else "todo",
             len(n_boxes[f]) if f in n_boxes else "",
             meta.get(f, {}).get("crop") or ""] for f in files]


def add_class(project, name: str) -> str:
    """Append a class to the project and save project.json.
    Never reorders. Returns the stored name; raises ValueError on bad/duplicate names."""
    name = re.sub(r"\s+", " ", (name or "").strip())
    if not name or "," in name:
        raise ValueError("class name must be non-empty and contain no comma")
    if name.lower() in (c.lower() for c in project.classes):
        raise ValueError(f"class '{name}' already exists")
    project.classes.append(name)
    project.save()
    return name


def add_image_field(project, name: str, kind: str = "str", unit: str = "", max_extra: int | None = None) -> Field:
    """Append an extra per-photo property (e.g. annotator, moisture) and save project.json.
    The key is *name* lower-cased with spaces -> ``_``. Raises ValueError on bad/duplicate names."""
    key = re.sub(r"\s+", "_", (name or "").strip().lower())
    if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
        raise ValueError("property name must start with a letter and use only letters, digits, spaces or _")
    if kind not in EXTRA_KINDS:
        raise ValueError(f"kind must be one of {EXTRA_KINDS}, got {kind!r}")
    taken = {f.key for f in project.image_fields()} | set(AUTO_IMAGE_KEYS) | {"file"}
    if key in taken:
        raise ValueError(f"property '{key}' already exists")
    if max_extra is not None and len(project.extra_image_fields) >= max_extra:
        raise ValueError(f"at most {max_extra} extra properties")
    project.extra_image_fields.append({"key": key, "kind": kind, "unit": (unit or "").strip()})
    project.save()
    return project.image_fields()[-1]


# ---- proposals combine / save ---------------------------------------------------------------------
MODES = ("replace model boxes (keep manual)", "replace all", "append")


def combine(existing: list[dict], new: list[dict], mode: str) -> list[dict]:
    if mode == "append":
        return [*existing, *new]
    if mode == "replace all":
        return list(new)
    return [b for b in existing if b.get("origin") == "manual"] + list(new)


def save_review(project, file_name: str, boxes: list[dict], meta_values: dict | list,
                proposer: str | None = None, reviewed: bool = True) -> dict:
    """Validate + write one photo through ``Store.upsert_image``. *meta_values* is a
    ``{field: value}`` dict or the form-ordered list. Boxes are in original pixels."""
    path = Path(project.photos_dir) / file_name
    with Image.open(path) as im:
        size = im.size
    store = project.store()
    prior = store.get_image(file_name)[1]
    fields = project.image_fields()
    values = meta_values if isinstance(meta_values, list) else [meta_values.get(f.key) for f in fields]
    meta, errors = form_to_meta(values, prior, fields)
    if errors:
        raise ValueError("; ".join(errors))
    clean = []
    for b in boxes:
        bb = clamp_box(b["bbox"], size)
        if bb is None:
            continue
        if b["class"] not in project.classes:
            raise ValueError(f"unknown class {b['class']!r}; add it as a new crop first")
        clean.append({**{k: b.get(k) for k in EXTRA_KEYS}, "bbox": bb, "class": b["class"],
                      "origin": b.get("origin") or "manual"})
    store.upsert_image(file_name, size[0], size[1], clean, meta, reviewed=reviewed, proposer=proposer)
    return {"file": file_name, "boxes": len(clean), "size": list(size)}
