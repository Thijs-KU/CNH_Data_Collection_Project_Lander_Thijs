"""Box drawing + an HTML contact sheet for quick visual review of annotations."""

from __future__ import annotations

import html
from pathlib import Path

from PIL import Image, ImageDraw

from . import coco as C

PALETTE = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 200, 255), (255, 0, 255),
           (255, 128, 0), (255, 255, 255), (0, 0, 0)]      # high contrast on wood / metal / grey floor


def color(i: int) -> tuple[int, int, int]:
    return PALETTE[i % len(PALETTE)]


def draw_boxes(img: Image.Image, boxes, out: str | Path, *, labels=None, scores=None,
               crop=None, width: int = 3, maxside: int | None = None, names=None,
               quality: int = 88) -> None:
    """Draw ``[x1,y1,x2,y2]`` boxes (colour per class id in *labels*) and save to *out*."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for i, b in enumerate(boxes):
        c = color(labels[i] if labels else 0)
        d.rectangle(list(b), outline=c, width=width)
        if scores is not None:
            d.text((b[0] + 2, max(0, b[1] - 12)), f"{scores[i]:.2f}", fill=c)
    if crop:
        im = im.crop(tuple(crop))
    if maxside:
        im.thumbnail((maxside, maxside))
    if names:  # legend
        d = ImageDraw.Draw(im)
        for i, n in enumerate(names):
            d.rectangle([4, 4 + 14 * i, 14, 14 + 14 * i], fill=color(i))
            d.text((18, 3 + 14 * i), n, fill=(255, 255, 255))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=quality) if str(out).lower().endswith((".jpg", ".jpeg")) else im.save(out)


def render_coco(coco: dict, images_dir: str | Path, out_dir: str | Path, *,
                crops: dict[str, list] | None = None, width: int = 3,
                maxside: int | None = None) -> list[dict]:
    """Render every image of *coco* with its boxes. Returns ``[{file, png, n}]`` rows."""
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    names = C.class_names(coco)
    rows = []
    for fname, items in C.boxes_by_file(coco).items():
        png = f"{Path(fname).stem}.png"
        with Image.open(images_dir / fname) as im:
            draw_boxes(im, [b for b, _ in items], out_dir / png, labels=[c for _, c in items],
                       crop=(crops or {}).get(fname), width=width, maxside=maxside, names=names)
        rows.append({"file": fname, "png": png, "n": len(items)})
    return rows


def write_index(path: str | Path, sections: dict[str, tuple[str, list[dict]]]) -> None:
    """``sections = {title: (relative_dir, rows)}`` -> a simple HTML grid."""
    parts = ["<!doctype html><meta charset=utf-8><title>annotations</title>"
             "<style>body{font:14px sans-serif;background:#111;color:#ddd}"
             "figure{display:inline-block;margin:6px;vertical-align:top}"
             "img{max-width:420px;display:block}</style>"]
    for title, (rel, rows) in sections.items():
        parts.append(f"<h2>{html.escape(title)} ({sum(r['n'] for r in rows)} boxes)</h2>")
        for r in rows:
            parts.append(f"<figure><a href='{rel}/{r['png']}'><img loading=lazy src='{rel}/{r['png']}'></a>"
                         f"<figcaption>{html.escape(r['file'])}: {r['n']}</figcaption></figure>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")
