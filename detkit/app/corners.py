"""Logic of the "Corners" tab (layout in corners_ui.py): list the project photos, mark the 4 corners of the A4
sheet on a raw one, warp it to the canvas of detkit/preprocess.py and replace the photo IN PLACE (same file name,
so its metadata stays attached). No gradio here."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from .. import coco as C
from ..preprocess import CANVAS_SIZE, SCALE_MM_PER_PX, WARPS_FILE, order_corners, warp_to_canvas
from . import logic as L

DISPLAY_MAXSIDE = 1600
COLUMNS = ["file", "warped", "status", "boxes", "size"]


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


def rows(project) -> list[list]:
    """The Review tab's photo list (same photos, same status / boxes) plus whether each is warped and its size."""
    warps = load_warps(project)
    out = []
    for f, status, boxes, _crop in L.photo_rows(L.list_photos(project.photos_dir), project.store()):
        try:
            with Image.open(project.photos_dir / f) as im:
                size = f"{im.width}x{im.height}"
        except OSError:
            size = "?"
        out.append([f, "warped" if f in warps else "raw", status, boxes, size])
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
    originals = project.workdir / "photos_original"
    if rec is None:
        originals.mkdir(parents=True, exist_ok=True)
        oname, k = name, 0
        while (originals / oname).exists():
            k += 1
            oname = f"{Path(name).stem}_{k}{Path(name).suffix}"
        shutil.copy2(raw, originals / oname)
    else:
        oname = rec["source"]
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
    warps[name] = {"source": oname, **info, **({"discarded": discarded} if discarded else {})}
    _warps_path(project).parent.mkdir(parents=True, exist_ok=True)
    _warps_path(project).write_text(json.dumps(warps, indent=1), encoding="utf-8")
    msg = f"Warped `{name}` in place ({CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}); the original is `photos_original/{oname}`."
    if old_boxes:
        msg += f" Its {len(old_boxes)} saved boxes were in the old frame and were cleared: review it again."
    return msg
