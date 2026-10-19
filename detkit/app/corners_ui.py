"""Gradio layout + events of the "Corners" tab (logic lives in corners.py)."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from ..preprocess import CANVAS_SIZE
from . import corners as K

HELP = (
    "### Corners: flatten the A4 sheet in a photo to {w}x{h}\n"
    "The list shows the same photos as the Review tab. Click a row, then click the **4 corners of the sheet** "
    "(any order); the warped result appears on the right once the 4th is set. *Undo* removes the last point, "
    "*Clear* all of them. **Warp this photo** replaces it in the project (same file name, same metadata); the raw "
    "photo is kept in `photos_original/` and you can redo the warp from it.\n\n"
    "Boxes saved for a photo are in its old frame, so warping clears them (a copy stays in `meta/warps.json`). "
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
    outs = [st, canvas, out, status]

    def refresh():
        return K.rows(project)

    refresh_b.click(refresh, outputs=[plist], api_name=False)
    if tab is not None:
        tab.select(refresh, outputs=[plist], api_name=False)

    def on_pick(evt: gr.SelectData):
        row = evt.row_value if evt.row_value else None
        return open_photo(row[0]) if row else [gr.skip()] * len(outs)

    plist.select(on_pick, outputs=outs, api_name=False)

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
