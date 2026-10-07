"""Cut large photos into overlapping square-ish tiles.

Why: the detector works on a fixed input size, so small objects (seeds) on a 12 MP photo would
shrink to a few pixels if the whole photo were resized. Tiles keep them large. The manifest records where each tile
came from so labels can be mapped back and train/val can be split by *source photo*.

``crops.json`` (optional) restricts tiling to a region of interest, resolved in order
``files[name]`` > orientation (``landscape``/``portrait``) > ``default``. Each entry is
``[x0, y0, x1, y1]`` or ``{"crop": [x0, y0, x1, y1], "tile": 900}``::

    {"default": null,
     "landscape": {"crop": [1200, 800, 3200, 2800], "tile": 800},
     "portrait":  {"crop": [600, 1100, 2200, 2600], "tile": 900},
     "files": {"IMG_0001.jpg": [0, 0, 2000, 2000]}}
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def axis_starts(lo: int, hi: int, tile: int, overlap: float) -> list[int]:
    """Start offsets along one axis covering [lo, hi) with tiles of size *tile*.

    Tiles overlap by at least *overlap* (fraction of tile) and the last tile ends at *hi*.
    """
    length = hi - lo
    if length <= tile:
        return [lo]
    max_step = tile * (1.0 - overlap)
    n = math.ceil((length - tile) / max_step) + 1
    step = (length - tile) / (n - 1)
    return [round(lo + i * step) for i in range(n)]


def plan_tiles(
    size: tuple[int, int], crop: tuple[int, int, int, int] | None, tile: int, overlap: float,
) -> list[tuple[int, int, int, int]]:
    """Return ``[(x, y, w, h)]`` tile rectangles for an image of *size* (W, H)."""
    W, H = size
    x0, y0, x1, y1 = crop if crop else (0, 0, W, H)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty crop {crop} for image size {size}")
    tw, th = min(tile, x1 - x0), min(tile, y1 - y0)
    return [
        (x, y, tw, th)
        for y in axis_starts(y0, y1, th, overlap)
        for x in axis_starts(x0, x1, tw, overlap)
    ]


def resolve_crop(name: str, size: tuple[int, int], crops: dict | None, tile: int):
    """Return ``(crop | None, tile)`` for one source image."""
    if not crops:
        return None, tile
    W, H = size
    entry = crops.get("files", {}).get(name)
    if entry is None:
        entry = crops.get("landscape" if W >= H else "portrait")
    if entry is None:
        entry = crops.get("default")
    if entry is None:
        return None, tile
    if isinstance(entry, dict):
        crop, tile = entry.get("crop"), entry.get("tile", tile)
    else:
        crop = entry
    return (tuple(crop) if crop else None), tile


def make_tiles(
    images_dir: str | Path, out_dir: str | Path, manifest_path: str | Path,
    tile: int = 800, overlap: float = 0.2, crops: dict | None = None, quality: int = 95,
) -> dict:
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not files:
        raise FileNotFoundError(f"no images in {images_dir}")
    man: dict = {"params": {"tile": tile, "overlap": overlap, "crops": crops},
                 "sources": {}, "tiles": []}
    for f in files:
        with Image.open(f) as im:
            im = im.convert("RGB")
            W, H = im.size
            crop, t = resolve_crop(f.name, (W, H), crops, tile)
            man["sources"][f.name] = {"width": W, "height": H,
                                      "crop": list(crop) if crop else None, "tile": t}
            for x, y, w, h in plan_tiles((W, H), crop, t, overlap):
                name = f"{f.stem}_x{x}_y{y}.jpg"
                im.crop((x, y, x + w, y + h)).save(out_dir / name, quality=quality)
                man["tiles"].append({"file": name, "source": f.name, "x": x, "y": y, "w": w, "h": h})
    Path(manifest_path).write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man
