"""Image preprocessing (OpenCV). Everything here is a no-op until you give the Preprocessor something to do.

Two kinds of preprocessing, applied in different places:

* ``canonicalize`` (geometric, e.g. warping the A4 sheet to a fixed top-down canvas). It may change the
  image size, so it runs ONCE, when a photo enters the project (``detkit import-photos``, the Review tab
  upload, the Inference tab upload). After that, the stored photo is the canonical one and every box,
  area feature, overlay and eval works in that frame without further changes.
* ``apply`` (photometric, e.g. contrast, white balance, shadow reduction). Must keep the image size.
  Not wired into the pipeline yet; intended for the top of ``predict.predict_image``.

Steps for ``apply`` are plain functions on an OpenCV image (BGR, uint8, HxWx3) that return the same shape::

    def clahe(bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    Preprocessor(steps=[clahe])

Manual A4 warp: the app's "Corners" tab (app/corners_ui.py) lists the project photos, lets you click the
4 sheet corners of a raw one and replaces it, in place, by the warped 900x1200 photo (``warp_to_canvas``
below). The raw photo moves to <workdir>/photos_original; which photos are warped is in meta/warps.json.
``import-photos`` and the Review upload copy photos as they are (raw): warp them afterwards in that tab.

A ``canonicalizer`` is a function ``PIL.Image (RGB) -> PIL.Image (RGB)``; set it on the module-level
``PREPROCESSOR`` below to switch the geometric stage on everywhere at once.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from PIL import Image

Step = Callable[[np.ndarray], np.ndarray]            # BGR uint8 HxWx3 -> BGR uint8 HxWx3, same size
Canonicalizer = Callable[[Image.Image], Image.Image]  # RGB PIL -> RGB PIL, size may change


class Preprocessor:
    def __init__(self, steps: Sequence[Step] | None = None, canonicalizer: Canonicalizer | None = None):
        self.steps: list[Step] = list(steps or [])
        self.canonicalizer = canonicalizer

    # ---- photometric, size-preserving (not wired in yet) ---------------------------------------
    def apply(self, image: Image.Image | np.ndarray) -> Image.Image | np.ndarray:
        """Run all steps in order. A PIL image (RGB) comes back as PIL RGB, a numpy array (BGR, as
        OpenCV reads it) comes back as numpy BGR. No steps: the input is returned as is."""
        if not self.steps:
            return image
        is_pil = isinstance(image, Image.Image)
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR) if is_pil else image
        size = bgr.shape[:2]
        for step in self.steps:
            bgr = step(bgr)
            if bgr.shape[:2] != size:
                raise ValueError(f"preprocessing step {getattr(step, '__name__', step)!r} changed the "
                                 f"image size {size} -> {bgr.shape[:2]}; use a canonicalizer for geometry")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) if is_pil else bgr

    __call__ = apply

    # ---- geometric, once per photo, when it enters the project ---------------------------------
    @property
    def changes_geometry(self) -> bool:
        return self.canonicalizer is not None

    def canonicalize(self, image: Image.Image) -> Image.Image:
        """Canonical version of a photo. For now: the image, unchanged."""
        return image if self.canonicalizer is None else self.canonicalizer(image)

    def ingest(self, src: str | Path, dst: str | Path) -> Path:
        """Put the source photo at ``dst`` in canonical form. While nothing changes the photo this is a
        byte-exact copy (so re-import checks that compare files keep working)."""
        src, dst = Path(src), Path(dst)
        if not self.changes_geometry:
            shutil.copy2(src, dst)
            return dst
        with Image.open(src) as im:                      # pixels are not EXIF-rotated, like the rest of detkit
            out = self.canonicalize(im.convert("RGB"))
        out.save(dst, quality=95)                        # re-encoded without EXIF; quality only affects JPEG
        return dst

    def canonical_path(self, src: str | Path, out_dir: str | Path) -> Path:
        """Path of the canonical version of ``src`` for read-only use (e.g. Inference uploads). Returns
        ``src`` itself while nothing changes the photo, otherwise writes it into ``out_dir`` under the same name."""
        src = Path(src)
        if not self.changes_geometry:
            return src
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return self.ingest(src, out_dir / src.name)


# One shared instance: set ``PREPROCESSOR.canonicalizer`` (or replace it) to turn the geometric stage on.
PREPROCESSOR = Preprocessor()


# ---- manual A4 warp --------------------------------------------------------------------------------
WARPS_FILE = "warps.json"                 # <workdir>/meta/: corners + matrix of every warped photo
CANVAS_SIZE = (900, 1200)                 # (width, height) in px of the warped photo: A4 portrait
SHEET_MM = (210.0, 297.0)                 # A4 (width, height) in mm, portrait
# 900x1200 is 3:4, A4 is 1:1.414, so the scale differs by axis (x: 0.2333, y: 0.2475 mm/px). Pixel AREAS
# still convert with one constant (mm2 = px2 * x * y), so ``scale_mm_per_px`` is their geometric mean.
MM_PER_PX = (SHEET_MM[0] / CANVAS_SIZE[0], SHEET_MM[1] / CANVAS_SIZE[1])
SCALE_MM_PER_PX = math.sqrt(MM_PER_PX[0] * MM_PER_PX[1])


def order_corners(pts) -> np.ndarray:
    """4 clicked points in any order -> float32 (4, 2) as top-left, top-right, bottom-right, bottom-left of
    the PORTRAIT sheet (short edge on top). Of the two rotations that qualify, the one whose first corner is
    nearer the image's top-left wins, so the warped photo comes out upright whatever the click order."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(p) != 4:
        raise ValueError(f"need exactly 4 corners, got {len(p)}")
    c = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]       # clockwise on screen (y points down)
    cross = []
    for i in range(4):
        a, b = p[(i + 1) % 4] - p[i], p[(i + 2) % 4] - p[(i + 1) % 4]
        cross.append(a[0] * b[1] - a[1] * b[0])
    if not (all(x > 0 for x in cross) or all(x < 0 for x in cross)):
        raise ValueError("the 4 points do not form a convex quadrilateral: click the corners of the sheet")
    best = None
    for k in range(4):
        q = np.roll(p, -k, axis=0)
        if np.linalg.norm(q[1] - q[0]) <= np.linalg.norm(q[3] - q[0]):     # top edge is the short edge
            if best is None or q[0].sum() < best[0]:
                best = (q[0].sum(), q)
    return best[1].astype(np.float32)


def warp_to_canvas(image: Image.Image, corners, size: tuple[int, int] = CANVAS_SIZE, supersample: int = 2
                   ) -> tuple[Image.Image, dict]:
    """Perspective-warp the quadrilateral ``corners`` (pixels of ``image``, any order) onto a ``size`` canvas.
    Warps at ``supersample`` x the size and shrinks with INTER_AREA, so a big photo does not alias. Returns
    the RGB image and ``{"corners" (ordered), "matrix" (source px -> canvas px), "canvas", "mm_per_px"}``."""
    W, H = size
    src = order_corners(corners)
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    def dst(k: int) -> np.ndarray:
        return np.float32([[0, 0], [W * k, 0], [W * k, H * k], [0, H * k]])

    big = cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(src, dst(supersample)),
                              (W * supersample, H * supersample), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    out = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA) if supersample > 1 else big
    info = {"corners": src.tolist(), "matrix": cv2.getPerspectiveTransform(src, dst(1)).tolist(),
            "canvas": [W, H], "mm_per_px": [SHEET_MM[0] / W, SHEET_MM[1] / H]}
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), info


# ---- brightness / contrast normalization ------------------------------------------------------------
TARGET_BRIGHTNESS = 230                   # default: median lightness (CIELAB L, 0-255) the photo is brought to
DEFAULT_CONTRAST = 1.0                    # default contrast enhancement (0 = off, 1 = full CLAHE)
MAX_GAMMA = 6.0                           # limits how far one photo may be pushed (gamma in [1/6, 6])


def _median_u8(a: np.ndarray) -> float:
    hist = np.bincount(a.ravel(), minlength=256)
    return float(np.searchsorted(np.cumsum(hist), a.size / 2))


def brightness(image: Image.Image, maxside: int = 512) -> float:
    """Median lightness (CIELAB L scaled to 0-255) of ``image``. After the A4 warp this is the lightness of the
    white paper (kernels cover well under half of the sheet), which is what ``normalize_brightness`` fixes."""
    im = image.convert("RGB")
    if max(im.size) > maxside:
        im.thumbnail((maxside, maxside), Image.BILINEAR)
    return _median_u8(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2LAB)[..., 0])


def normalize_brightness(image: Image.Image, target: float = TARGET_BRIGHTNESS, contrast: float = DEFAULT_CONTRAST
                         ) -> tuple[Image.Image, dict]:
    """Bring the median lightness of ``image`` to ``target`` (1-254), optionally with contrast enhancement.

    * ``contrast`` 0-1: how much of a CLAHE (local histogram equalisation, clip 2.0, 8x8 tiles) of the lightness
      channel is mixed in BEFORE the brightness step. 0 = none. Colours (a, b channels) are not touched.
    * brightness: one gamma curve on the lightness, chosen so the median lands on ``target``. A gamma keeps 0
      and 255 fixed, so nothing clips; the gain is limited to gamma in [1/6, 6], so ``after`` can miss ``target``.
    The size never changes, so boxes stay valid. Returns the RGB image and
    ``{"target", "contrast", "gamma", "before", "after"}``."""
    if not 1 <= target <= 254:
        raise ValueError(f"target brightness must be between 1 and 254, got {target}")
    if not 0 <= contrast <= 1:
        raise ValueError(f"contrast must be between 0 and 1, got {contrast}")
    lab = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB)
    lum = np.ascontiguousarray(lab[..., 0])
    before = _median_u8(lum)
    if contrast > 0:
        eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lum)
        lum = cv2.addWeighted(lum, 1 - contrast, eq, contrast, 0)
    mid = min(max(_median_u8(lum), 5.0), 250.0)                      # keep the logs below finite and non-zero
    gamma = float(np.clip(math.log(target / 255) / math.log(mid / 255), 1 / MAX_GAMMA, MAX_GAMMA))
    for _ in range(3):                                               # converting back to RGB can clip saturated
        lut = np.clip(np.round(255 * (np.arange(256) / 255) ** gamma), 0, 255).astype(np.uint8)   # colours, so
        lab[..., 0] = cv2.LUT(lum, lut)                              # measure the real output and correct
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        after = _median_u8(np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]))
        if abs(after - target) <= 1 or not 0 < after < 255:
            break
        gamma = float(np.clip(gamma * math.log(target / 255) / math.log(after / 255), 1 / MAX_GAMMA, MAX_GAMMA))
    return Image.fromarray(rgb), {"target": float(target), "contrast": float(contrast), "gamma": round(gamma, 4),
                                  "before": before, "after": after}


# ---- the same normalization for photos that are NOT in the project yet ---------------------------------
def in_project_photos(path: str | Path) -> bool:
    """Is ``path`` a photo inside a project's ``photos`` folder (<workdir>/photos next to project.json)? Those are
    already preprocessed (warped / normalized in the Preprocessing tab) and must not be processed twice."""
    d = Path(path).resolve().parent
    return d.name == "photos" and (d.parent / "project.json").exists()


def prepare_photo(image: Image.Image, source: str | Path, normalize: tuple[float, float] | None
                  ) -> tuple[Image.Image, dict | None]:
    """Detection-time brightness / contrast normalization of a RAW photo, with the same function the
    Preprocessing tab applies to project photos, so uploads, ``detkit predict`` and ``detkit eval --images``
    are comparable with them. ``normalize`` = (target, contrast), or None for no change. Photos that already are
    in a project's ``photos`` folder are returned unchanged. Returns the image and the info dict (None if untouched)."""
    if not normalize or in_project_photos(source):
        return image, None
    return normalize_brightness(image, *normalize)


def before_after(before: Image.Image, after: Image.Image, info: dict, name: str = "", height: int = 500) -> Image.Image:
    """Original and normalized photo side by side with their median brightness, to look at what was changed."""
    from PIL import ImageDraw

    def small(im):
        k = height / im.height
        return im.convert("RGB").resize((max(1, round(im.width * k)), height), Image.LANCZOS)

    a, b, bar = small(before), small(after), 22
    out = Image.new("RGB", (a.width + b.width + 6, height + bar), (30, 30, 30))
    out.paste(a, (0, bar))
    out.paste(b, (a.width + 6, bar))
    d = ImageDraw.Draw(out)
    d.text((4, 5), f"{name} original  brightness {info['before']:.0f}", fill=(255, 255, 255))
    d.text((a.width + 10, 5), f"normalized -> {info['after']:.0f} (target {info['target']:.0f}, "
                              f"contrast {info['contrast']:.2f})", fill=(255, 255, 255))
    return out
