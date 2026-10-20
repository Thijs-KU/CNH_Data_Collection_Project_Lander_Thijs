"""Logic of the "Preprocessing" tab (layout in corners_ui.py; the file keeps its old name): list the project
photos, then (1) mark the 4 corners of the A4 sheet on a raw one and warp it to the canvas of
detkit/preprocess.py, and/or (2) normalize its brightness / contrast. Both replace the photo IN PLACE (same
file name, so its metadata stays attached); the raw photo is kept in photos_original. No gradio here.

meta/warps.json has one record per changed photo: ``source`` (file in photos_original), the warp
(``corners``, ``matrix``, ... only if it was warped) and ``normalize`` (only if its brightness was changed).
The stored photo is always original -> recorded warp -> recorded brightness, so any step can be redone."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from .. import coco as C
from ..preprocess import (CANVAS_SIZE, SCALE_MM_PER_PX, TARGET_BRIGHTNESS, WARPS_FILE, brightness,
                          normalize_brightness, order_corners, warp_to_canvas)
from . import logic as L

DISPLAY_MAXSIDE = 1600
COLUMNS = ["file", "warped", "status", "boxes", "size", "normalized", "brightness"]


# ---- what is warped ------------------------------------------------------------------------------------
def _warps_path(project) -> Path:
    return project.workdir / "meta" / WARPS_FILE


def load_warps(project) -> dict:
    f = _warps_path(project)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def source_of(project, name: str) -> Path:
    """The RAW photo to click on: the saved original if ``name`` was warped, else the project photo itself."""
    rec = load_warps(project).get(name)
    return project.workdir / "photos_original" / rec["source"] if rec else project.photos_dir / name


def is_warped(rec: dict | None) -> bool:
    return bool(rec and rec.get("corners"))


def rows(project) -> list[list]:
    """The Review tab's photo list (same photos, same status / boxes) plus whether each is warped, its size,
    whether its brightness was normalized and its current median lightness (0-255)."""
    warps = load_warps(project)
    out = []
    for f, status, boxes, _crop in L.photo_rows(L.list_photos(project.photos_dir), project.store()):
        rec = warps.get(f)
        try:
            with Image.open(project.photos_dir / f) as im:
                size, bright = f"{im.width}x{im.height}", round(brightness(im))
        except OSError:
            size, bright = "?", ""
        out.append([f, "warped" if is_warped(rec) else "raw", status, boxes, size,
                    "yes" if rec and rec.get("normalize") else "-", bright])
    return out


# ---- clicking --------------------------------------------------------------------------------------------
def new_state() -> dict:
    """``pts`` are in pixels of the RAW photo ``path``; ``scale`` = raw / displayed size."""
    return {"name": None, "path": None, "scale": 1.0, "pts": []}


def load_display(path: str | Path, maxside: int = DISPLAY_MAXSIDE) -> tuple[Image.Image, float]:
    """RGB copy of the photo for clicking, longest side <= ``maxside``, and ``scale`` (raw / shown).
    NOT EXIF-rotated: detkit reads pixels as stored, and so does the warp."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = max(1.0, max(im.size) / maxside)
        if scale > 1:
            im = im.resize((round(im.width / scale), round(im.height / scale)), Image.LANCZOS)
    return im, scale


def draw_points(display: Image.Image, pts_shown: list[list[float]]) -> Image.Image:
    """Copy of ``display`` with the clicked corners numbered and joined (closed once there are 4)."""
    im = display.copy()
    d = ImageDraw.Draw(im)
    r = max(6, max(im.size) // 120)
    xy = [tuple(p) for p in pts_shown]
    if len(xy) == 4:                                    # outline in a convex order, whatever the click order
        cx, cy = sum(x for x, _ in xy) / 4, sum(y for _, y in xy) / 4
        ring = sorted(xy, key=lambda q: math.atan2(q[1] - cy, q[0] - cx))
        d.line(ring + [ring[0]], fill=(255, 40, 40), width=max(2, r // 3))
    elif len(xy) > 1:
        d.line(xy, fill=(255, 40, 40), width=max(2, r // 3))
    for i, (x, y) in enumerate(xy, 1):
        d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 0), width=max(2, r // 3))
        d.text((x + r + 2, y - r), str(i), fill=(255, 255, 0))
    return im


def shown_points(state: dict) -> list[list[float]]:
    return [[x / state["scale"], y / state["scale"]] for x, y in state["pts"]]


def add_point(state: dict, x_shown: float, y_shown: float) -> dict:
    """Click at (x, y) of the displayed image. A 5th click is ignored (undo / clear first)."""
    s = dict(state)
    if len(s["pts"]) < 4:
        s["pts"] = [*s["pts"], [x_shown * s["scale"], y_shown * s["scale"]]]
    return s


def undo(state: dict) -> dict:
    return {**state, "pts": state["pts"][:-1]}


def clear(state: dict) -> dict:
    return {**state, "pts": []}


def preview(state: dict) -> Image.Image | None:
    """The warped photo once 4 corners are set, else None."""
    if len(state["pts"]) != 4 or not state["path"]:
        return None
    with Image.open(state["path"]) as im:
        return warp_to_canvas(im, state["pts"])[0]


# ---- keeping the original ----------------------------------------------------------------------------------
def _keep_original(project, name: str, raw: Path, rec: dict | None) -> str:
    """Name of the saved original of ``name`` in photos_original; copies ``raw`` there the first time."""
    if rec is not None:
        return rec["source"]
    originals = project.workdir / "photos_original"
    originals.mkdir(parents=True, exist_ok=True)
    oname, k = name, 0
    while (originals / oname).exists():
        k += 1
        oname = f"{Path(name).stem}_{k}{Path(name).suffix}"
    shutil.copy2(raw, originals / oname)
    return oname


def _write_warps(project, warps: dict) -> None:
    _warps_path(project).parent.mkdir(parents=True, exist_ok=True)
    _warps_path(project).write_text(json.dumps(warps, indent=1), encoding="utf-8")


# ---- warping ----------------------------------------------------------------------------------------------
def warp_photo(project, state: dict) -> str:
    """Replace the project photo ``state["name"]`` by its warped version (same file name).

    * The raw photo is kept in ``<workdir>/photos_original`` (first warp only); a redo starts from that original.
    * Its boxes are in the old frame, so they are cleared and it becomes unreviewed; the rest of its metadata
      (crop, pile_id, weight, ...) stays. The discarded boxes are kept in ``meta/warps.json`` under "discarded".
    * ``meta/warps.json`` records the corners and matrix.
    Returns a message for the user."""
    name = state["name"]
    if not name or len(state["pts"]) != 4:
        raise ValueError("pick a photo and mark all 4 corners of the sheet first")
    dst = project.photos_dir / name
    warps = load_warps(project)
    rec = warps.get(name)
    raw = source_of(project, name)
    with Image.open(raw) as im:                                         # nothing is touched if this fails
        warped, info = warp_to_canvas(im, state["pts"])
        norm = (rec or {}).get("normalize")
        if norm:                                                        # a redone warp keeps its brightness settings
            warped, norm = normalize_brightness(warped, norm["target"], norm["contrast"])
    oname = _keep_original(project, name, raw, rec)
    warped.save(dst, quality=95)                                        # quality only matters for JPEG

    store = project.store()
    old_boxes, old_meta = store.get_image(name)
    meta = {"scale_mm_per_px": round(SCALE_MM_PER_PX, 6)} if "scale_mm_per_px" in store.image_keys else {}
    store.upsert_image(name, *CANVAS_SIZE, [], meta, reviewed=False)    # clears the boxes, keeps other metadata
    coco, all_meta = store.load_coco(), store.load_meta()
    for im in coco["images"]:                                           # upsert does not update the size of a known photo
        if im["file_name"] == name:
            im["width"], im["height"] = CANVAS_SIZE
    store.save(coco, all_meta)

    discarded = list((rec or {}).get("discarded", []))
    if old_boxes or old_meta.get("reviewed"):
        discarded.append({"reviewed": bool(old_meta.get("reviewed")), "boxes": old_boxes})
    warps[name] = {"source": oname, **info, **({"normalize": norm} if norm else {}),
                   **({"discarded": discarded} if discarded else {})}
    _write_warps(project, warps)
    msg = f"Warped `{name}` in place ({CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}); the original is `photos_original/{oname}`."
    if old_boxes:
        msg += f" Its {len(old_boxes)} saved boxes were in the old frame and were cleared: review it again."
    return msg


# ---- brightness / contrast -------------------------------------------------------------------------------
def _render(project, name: str, normalize: tuple[float, float] | None) -> tuple[Image.Image, dict | None]:
    """The photo as it would be stored: original -> its recorded warp (if any) -> brightness ``normalize`` =
    (target, contrast), or none. Returns the image and the brightness info (None without ``normalize``)."""
    rec = load_warps(project).get(name)
    with Image.open(source_of(project, name)) as im:
        im = im.convert("RGB")                                          # not EXIF-rotated, like the stored frame
        if is_warped(rec):
            im = warp_to_canvas(im, rec["corners"])[0]
        return normalize_brightness(im, *normalize) if normalize else (im, None)


def normalized_preview(project, name: str | None, target: float, contrast: float) -> tuple[Image.Image | None, str]:
    """What *Apply* would store for ``name``, and a one-line summary."""
    if not name or not (project.photos_dir / name).exists():
        return None, ""
    img, info = _render(project, name, (target, contrast))
    return img, (f"Median brightness {info['before']:.0f} -> {info['after']:.0f} (target {target:.0f}, "
                 f"gamma {info['gamma']:.2f}" + (f", contrast {contrast:.2f}" if contrast else "") + ").")


def normalize_photo(project, name: str, target: float = TARGET_BRIGHTNESS, contrast: float = 0.0) -> str:
    """Store ``name`` with its median brightness at ``target`` (and ``contrast`` 0-1 enhancement), in place.
    Always computed from the original (and the recorded warp), so repeating it with other settings does not
    stack. Boxes are NOT touched: brightness does not move pixels."""
    dst = project.photos_dir / name
    if not dst.exists():
        raise ValueError(f"`{name}` is not in the project")
    warps = load_warps(project)
    rec = warps.get(name)
    raw = source_of(project, name)
    img, info = _render(project, name, (target, contrast))              # nothing is touched if this fails
    oname = _keep_original(project, name, raw, rec)
    img.save(dst, quality=95)
    warps[name] = {**(rec or {"source": oname}), "normalize": info}
    _write_warps(project, warps)
    return f"`{name}`: median brightness {info['before']:.0f} -> {info['after']:.0f}."


def reset_photo_brightness(project, name: str) -> str:
    """Undo the brightness change of ``name`` (its warp stays), again from the original."""
    warps = load_warps(project)
    rec = warps.get(name)
    if not rec or not rec.get("normalize"):
        return f"`{name}` has no brightness change to remove."
    img, _ = _render(project, name, None)
    img.save(project.photos_dir / name, quality=95)
    warps[name] = {k: v for k, v in rec.items() if k != "normalize"}
    _write_warps(project, warps)
    return f"Removed the brightness change from `{name}`."


def normalize_all(project, target: float = TARGET_BRIGHTNESS, contrast: float = 0.0) -> str:
    """:func:`normalize_photo` for every photo of the project, so they all end at the same brightness."""
    names, done, failed = L.list_photos(project.photos_dir), 0, []
    for n in names:
        try:
            normalize_photo(project, n, target, contrast)
            done += 1
        except (ValueError, OSError) as e:
            failed.append(f"{n} ({e})")
    msg = f"Normalized {done} of {len(names)} photos to brightness {target:.0f}."
    return msg + (f" Failed: {', '.join(failed)}." if failed else "")
