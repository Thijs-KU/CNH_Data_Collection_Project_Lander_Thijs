"""Gradio layout + events of the "Preprocessing" tab: A4 warp and brightness / contrast (logic: corners.py)."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from ..preprocess import CANVAS_SIZE, TARGET_BRIGHTNESS
from . import corners as K

HELP = (
    "### Preprocessing: (1) flatten the A4 sheet to {w}x{h}, (2) bring the brightness to a fixed level\n"
    "The list shows the same photos as the Review tab. **Step 1, warp:** click a row, then click the **4 corners of the sheet** "
    "(any order); the warped result appears on the right once the 4th is set. *Undo* removes the last point, "
    "*Clear* all of them. **Warp this photo** replaces it in the project (same file name, same metadata); the raw "
    "photo is kept in `photos_original/` and you can redo the warp from it.\n\n"
    "Boxes saved for a photo are in its old frame, so warping clears them (a copy stays in `meta/warps.json`). "
    "**Step 2, brightness** (below the warp preview): set the target brightness and the contrast enhancement, "
    "check the preview, then apply it to this photo or to all photos. Brightness is measured on the paper "
    "(the median over the photo), so warp first. It keeps your boxes and is always redone from the saved original, "
    "so you can change the settings later. "
    "New raw photos: add them in the Review tab first, then warp them here. Add an extra property named "
    "`scale_mm_per_px` (Review tab) to get the area features in mm². Reload the list in the Review tab afterwards."
).format(w=CANVAS_SIZE[0], h=CANVAS_SIZE[1])


def build_corners_tab(project, wd: Path, tab=None) -> None:
    shown: dict = {}                                    # path -> (display image, scale): one entry, the open photo

    def display_of(s: dict):
        if s["path"] not in shown:
            shown.clear()
            shown[s["path"]] = K.load_display(s["path"])
        return shown[s["path"]][0]

    def view(s: dict, msg: str = ""):
        if not s["path"]:
            return s, None, None, msg
        img = K.draw_points(display_of(s), K.shown_points(s))
        n = len(s["pts"])
        return s, img, K.preview(s), msg or (f"{n}/4 corners set." if n < 4 else "4 corners set: check the preview, "
                                                                                 "then warp the photo.")

    def open_photo(name: str):
        path = K.source_of(project, name)
        if not path.exists():
            return view(K.new_state(), f"`{path}` not found.")
        s = {**K.new_state(), "name": name, "path": str(path)}
        display_of(s)
        s["scale"] = shown[s["path"]][1]
        row = next((r for r in K.rows(project) if r[0] == name), None)
        note = "Already warped: mark the corners again (on the original) to redo it. " if row and row[1] == "warped" else ""
        if row and (row[3] or row[2] == "reviewed"):
            note += f"Warping clears its {row[3] or 0} saved boxes (a copy is kept in meta/warps.json). "
        return view(s, note + "Click the 4 corners of the sheet.")

    gr.Markdown(HELP)
    st = gr.State(K.new_state())
    with gr.Row():
        with gr.Column(scale=3):
            plist = gr.Dataframe(headers=K.COLUMNS, value=K.rows(project), interactive=False, max_height=240,
                                 label="photos in the project (click a row to open)")
            refresh_b = gr.Button("Refresh list")
            canvas = gr.Image(type="pil", interactive=False, label="Click the 4 corners of the sheet")
            with gr.Row():
                undo_b = gr.Button("Undo last point")
                clear_b = gr.Button("Clear points")
        with gr.Column(scale=2):
            out = gr.Image(type="pil", interactive=False, label=f"Warped {CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}")
            warp_b = gr.Button("Warp this photo", variant="primary")
            status = gr.Markdown()
            gr.Markdown("#### Brightness and contrast")
            target = gr.Slider(60, 240, value=TARGET_BRIGHTNESS, step=1, label="Target brightness (median, 0-255)",
                               info="all photos end at this level; ~200 = light grey-white paper")
            contrast = gr.Slider(0, 1, value=0.0, step=0.05, label="Contrast enhancement (0 = off)",
                                 info="mixes in local contrast (CLAHE) before the brightness step")
            bprev = gr.Image(type="pil", interactive=False, label="Brightness preview (what Apply stores)")
            bstatus = gr.Markdown()
            with gr.Row():
                norm_b = gr.Button("Apply to this photo", variant="primary")
                norm_all_b = gr.Button("Apply to all photos")
            reset_b = gr.Button("Remove brightness change from this photo")
    outs = [st, canvas, out, status]

    def refresh():
        return K.rows(project)

    refresh_b.click(refresh, outputs=[plist], api_name=False)
    if tab is not None:
        tab.select(refresh, outputs=[plist], api_name=False)

    def on_pick(evt: gr.SelectData):
        row = evt.row_value if evt.row_value else None
        return open_photo(row[0]) if row else [gr.skip()] * len(outs)

    def on_click(s, evt: gr.SelectData):
        if not s["path"]:
            return view(s, "Click a photo in the list first.")
        if len(s["pts"]) >= 4:
            return view(s, "4 corners are already set: Undo or Clear to change them.")
        x, y = evt.index
        return view(K.add_point(s, x, y))

    canvas.select(on_click, inputs=[st], outputs=outs, api_name=False)
    undo_b.click(lambda s: view(K.undo(s)), inputs=[st], outputs=outs, api_name=False)
    clear_b.click(lambda s: view(K.clear(s)), inputs=[st], outputs=outs, api_name=False)

    def warp(s):
        try:
            msg = K.warp_photo(project, s)
        except ValueError as e:
            raise gr.Error(str(e))
        s = {**s, "path": str(K.source_of(project, s["name"]))}          # a redo clicks on the saved original
        return s, msg + " Reload the photo list in the Review tab.", K.rows(project)

    warp_b.click(warp, inputs=[st], outputs=[st, status, plist], api_name=False)

    # ---- brightness / contrast
    def bpreview(s, tgt, con):
        try:
            img, msg = K.normalized_preview(project, s["name"], tgt, con)
        except (ValueError, OSError) as e:
            return None, str(e)
        return img, msg

    plist.select(on_pick, outputs=outs, api_name=False).then(bpreview, inputs=[st, target, contrast],
                                                              outputs=[bprev, bstatus], api_name=False)
    target.release(bpreview, inputs=[st, target, contrast], outputs=[bprev, bstatus], api_name=False)
    contrast.release(bpreview, inputs=[st, target, contrast], outputs=[bprev, bstatus], api_name=False)

    def need(s):
        if not s["name"]:
            raise gr.Error("click a photo in the list first")
        return s["name"]

    def apply_one(s, tgt, con):
        try:
            msg = K.normalize_photo(project, need(s), tgt, con)
        except ValueError as e:
            raise gr.Error(str(e))
        return msg + " Reload the photo list in the Review tab.", K.rows(project)

    def apply_all(tgt, con):
        return K.normalize_all(project, tgt, con) + " Reload the photo list in the Review tab.", K.rows(project)

    def reset_one(s):
        return K.reset_photo_brightness(project, need(s)), K.rows(project)

    for btn, fn, ins in ((norm_b, apply_one, [st, target, contrast]), (norm_all_b, apply_all, [target, contrast]),
                         (reset_b, reset_one, [st])):
        btn.click(fn, inputs=ins, outputs=[status, plist], api_name=False).then(
            bpreview, inputs=[st, target, contrast], outputs=[bprev, bstatus], api_name=False)
