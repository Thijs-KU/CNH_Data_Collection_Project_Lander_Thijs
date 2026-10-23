"""Gradio layout + events of the "Inference" tab (logic lives in inference.py)."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

from ..preprocess import CANVAS_SIZE, DEFAULT_CONTRAST, TARGET_BRIGHTNESS
from ..project import DEFAULT_SCORE, DEFAULT_TILE
from . import inference as I
from . import corners as K
from . import proposals as P

PHOTO_COLS = ["file", "crop", "total_weight_g"]

WARP_HELP = (
    "New photos that are not in the project yet can be flattened the same way as in the Preprocessing tab: pick an "
    "uploaded photo, click the **4 corners of the A4 sheet** (any order), check the warped preview, then "
    "**Use this warp**. The warp is applied (before the brightness step) when you run inference. *Save as samples* "
    "stores the warped photo and keeps the raw upload in `photos_original/`, so the Preprocessing tab can redo it. "
    "Photos without a warp are used as they are. The ROI below is in pixels of the photo after the warp."
)

TODO_TEXT = (
    "**Student tasks** - `low_g` / `high_g` (95 % prediction interval of each estimate) appear once the constants "
    "are calibrated on at least 2 weighed piles per crop (`detkit weight calibrate`, tasks 2 and 3); "
    "editing a constant by hand removes its bounds. "
    "See `docs/STUDENT_TASKS.md`."
)


def _df(rows, cols):
    return pd.DataFrame(rows, columns=cols)


def _clean(v):
    return None if v is None or (isinstance(v, float) and v != v) or v == "" else v


def per_photo_from_table(df, classes: list[str]) -> dict[str, dict]:
    """Editable per-photo table -> ``{file: {crop, total_weight_g}}`` (validated)."""
    out = {}
    for r in (df.to_dict(orient="records") if df is not None else []):
        crop = _clean(r.get("crop"))
        if crop is not None and crop not in classes:
            raise ValueError(f"{r['file']}: crop '{crop}' is not one of {classes}")
        out[str(r["file"])] = {"crop": crop, "total_weight_g": _clean(r.get("total_weight_g"))}
    return out


def build_inference_tab(project, wd: Path, ckpt: str | None, device: str) -> None:
    cache = wd / ".app_cache" / "inference"
    best, best_note = I.default_score(wd)
    score0 = DEFAULT_SCORE
    score_note = f"default {DEFAULT_SCORE}" + (f"; {best_note}: {best}" if best_note.startswith("best-F1") else "")
    cks = P.find_checkpoints(wd, ckpt, project.proposer_ckpt)
    d0 = I.ckpt_defaults(cks[0] if cks else None, project)
    classes = project.classes

    gr.Markdown("### Inference: photos -> boxes -> kernel count / area -> weight")
    gr.Markdown(TODO_TEXT)
    with gr.Row():
        with gr.Column(scale=2):
            files = gr.File(file_count="multiple", file_types=["image"], type="filepath",
                            label="Upload one or more photos")
            photo_tbl = gr.Dataframe(
                headers=PHOTO_COLS, datatype=["str", "str", "number"], interactive=True,
                static_columns=[0], row_count=(0, "fixed"), value=_df([], PHOTO_COLS), max_height=220,
                label=f"Optional per photo: crop ({', '.join(classes)}), measured total weight (g)")
        with gr.Column(scale=2):
            ck_dd = gr.Dropdown(choices=cks, value=cks[0] if cks else None, allow_custom_value=True,
                                label="RF-DETR checkpoint")
            model_dd = gr.Dropdown(choices=I.model_choices(), value="count_x_constant", label="weight model")
            with gr.Row():
                score = gr.Slider(0.05, 0.95, score0, step=0.01, label="score threshold", info=score_note)
                tile = gr.Slider(200, 1600, DEFAULT_TILE, step=50, label="tile (px)")
                overlap = gr.Slider(0.0, 0.5, d0["overlap"], step=0.05, label="overlap")
            with gr.Row():
                max_side = gr.Slider(0, 6000, 0, step=100, label="downscale longest side to (0 = off; faster on CPU)")
                roi = gr.Textbox(label="ROI x0,y0,x1,y1 (px of the photo after warping, optional)")
            show_kernel = gr.Checkbox(True, label="Show the kernel pixels (green) inside the detected boxes",
                                      info="the pixels the kernel-area features count; box background stays untinted")
            norm_on = gr.Checkbox(True, label="Normalize brightness / contrast first (same as the Preprocessing tab)",
                                  info="uploads are made as bright as the project photos; photos already in the "
                                       "project are not changed again")
            with gr.Row():
                n_target = gr.Slider(60, 240, TARGET_BRIGHTNESS, step=1, label="target brightness (median, 0-255)")
                n_contrast = gr.Slider(0, 1, DEFAULT_CONTRAST, step=0.05, label="contrast enhancement (0 = off)")
            run_b = gr.Button("Run inference", variant="primary")
    corners_st = gr.State({})                       # uploaded file name -> its 4 sheet corners (px of the upload)
    w_state = gr.State(K.new_state())
    w_shown: dict = {}                              # path -> (display image, scale): the photo that is open

    with gr.Accordion("Warp new photos to the A4 sheet (optional)", open=False):
        gr.Markdown(WARP_HELP)
        with gr.Row():
            with gr.Column(scale=3):
                w_pick = gr.Dropdown(choices=[], value=None, label="uploaded photo")
                w_canvas = gr.Image(type="pil", interactive=False, label="Click the 4 corners of the sheet")
                with gr.Row():
                    w_undo = gr.Button("Undo last point")
                    w_clear = gr.Button("Clear points")
            with gr.Column(scale=2):
                w_out = gr.Image(type="pil", interactive=False, label=f"Warped {CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}")
                with gr.Row():
                    w_use = gr.Button("Use this warp", variant="primary")
                    w_drop = gr.Button("Remove the warp of this photo")
                w_status = gr.Markdown()
    pre_gal = gr.Gallery(label="Preprocessing: original (left) vs normalized (right) - what the detector sees",
                         columns=1, height=420, object_fit="contain")
    status = gr.Markdown("")
    gallery = gr.Gallery(label="Detections", columns=4, height=340, object_fit="contain")
    res = gr.Dataframe(headers=I.RESULT_COLUMNS, value=_df([], I.RESULT_COLUMNS), interactive=False,
                       wrap=True, max_height=300, label="Results (weight from the selected model)")
    with gr.Row():
        csv_f = gr.File(label="Download results (CSV)", interactive=False)
        save_b = gr.Button("Save as samples (photos + boxes go to the Review tab, unreviewed)")
    items = gr.State([])

    with gr.Accordion("Model constants (weight_config.json)", open=False):
        const_tbl = gr.Dataframe(headers=I.CONST_COLUMNS, datatype=["str", "number", "number"],
                                 value=_df(I.constants_rows(project), I.CONST_COLUMNS),
                                 interactive=True, static_columns=[0], row_count=(0, "fixed"),
                                 label="per crop: grams per kernel (count_x_constant), grams per px^2 of box "
                                       "area (area_x_constant)")
        const_b = gr.Button("Save constants")
        const_msg = gr.Markdown("")

    # ---- events ------------------------------------------------------------------------------
    def on_upload(paths):
        rows = [[Path(p).name, "", None] for p in (paths or [])]
        return _df(rows, PHOTO_COLS)

    files.change(on_upload, inputs=[files], outputs=[photo_tbl], api_name=False)

    def preview(paths, on, tgt, con, corners):
        """Show the warp and the normalization before anything is detected (no model needed)."""
        ps = [Path(p) for p in (paths or [])]
        if not ps or not (on or corners):
            return []
        try:
            return I.prepare_photos(ps, project, cache, (float(tgt), float(con)) if on else None, corners or {})[1]
        except (ValueError, OSError) as e:
            raise gr.Error(str(e))

    pre_in = [files, norm_on, n_target, n_contrast, corners_st]

    # ---- warp new photos (corner clicking as in the Preprocessing tab)
    def uploaded(paths) -> dict[str, str]:
        return {Path(p).name: str(p) for p in (paths or [])}

    def w_display(s: dict):
        if s["path"] not in w_shown:
            w_shown.clear()
            w_shown[s["path"]] = K.load_display(s["path"])
        return w_shown[s["path"]][0]

    def w_view(s: dict, msg: str = ""):
        if not s["path"]:
            return s, None, None, msg
        n = len(s["pts"])
        return (s, K.draw_points(w_display(s), K.shown_points(s)), K.preview(s),
                msg or (f"{n}/4 corners set." if n < 4 else "4 corners set: check the preview, then use the warp."))

    def w_open(name, paths, corners):
        path = uploaded(paths).get(name)
        if not path:
            return w_view(K.new_state())
        s = {**K.new_state(), "name": name, "path": path}
        w_display(s)
        s["scale"] = w_shown[path][1]
        saved = (corners or {}).get(name)
        if saved:
            s["pts"] = [[float(x), float(y)] for x, y in saved]
        return w_view(s, "This photo has a warp: Clear the points to redo it." if saved
                      else "Click the 4 corners of the sheet.")

    w_outs = [w_state, w_canvas, w_out, w_status]

    def w_on_upload(paths, corners):
        names = list(uploaded(paths))
        keep = {n: c for n, c in (corners or {}).items() if n in names}      # drop warps of removed uploads
        return gr.update(choices=names, value=names[0] if names else None), keep

    files.change(w_on_upload, inputs=[files, corners_st], outputs=[w_pick, corners_st], api_name=False).then(
        w_open, inputs=[w_pick, files, corners_st], outputs=w_outs, api_name=False)
    w_pick.change(w_open, inputs=[w_pick, files, corners_st], outputs=w_outs, api_name=False)

    def w_click(s, evt: gr.SelectData):
        if not s["path"]:
            return w_view(s, "Upload a photo and pick it first.")
        if len(s["pts"]) >= 4:
            return w_view(s, "4 corners are already set: Undo or Clear to change them.")
        x, y = evt.index
        return w_view(K.add_point(s, x, y))

    w_canvas.select(w_click, inputs=[w_state], outputs=w_outs, api_name=False)
    w_undo.click(lambda s: w_view(K.undo(s)), inputs=[w_state], outputs=w_outs, api_name=False)
    w_clear.click(lambda s: w_view(K.clear(s)), inputs=[w_state], outputs=w_outs, api_name=False)

    def w_apply(s, corners):
        if not s["name"] or len(s["pts"]) != 4:
            raise gr.Error("pick a photo and mark all 4 corners of the sheet first")
        c = dict(corners or {})
        c[s["name"]] = [[float(x), float(y)] for x, y in s["pts"]]
        return c, f"Warp set for `{s['name']}` ({len(c)} photo(s) with a warp); it is applied when you run inference."

    def w_remove(s, corners):
        c = dict(corners or {})
        had = c.pop(s["name"], None) is not None
        s2, img, out, _msg = w_view(K.clear(s))
        return c, s2, img, out, (f"Warp of `{s['name']}` removed." if had else "This photo has no warp.")

    w_use.click(w_apply, inputs=[w_state, corners_st], outputs=[corners_st, w_status], api_name=False).then(
        preview, inputs=pre_in, outputs=[pre_gal], api_name=False)
    w_drop.click(w_remove, inputs=[w_state, corners_st], outputs=[corners_st, *w_outs], api_name=False).then(
        preview, inputs=pre_in, outputs=[pre_gal], api_name=False)

    for trig in (files.change, norm_on.change, n_target.release, n_contrast.release):
        trig(preview, inputs=pre_in, outputs=[pre_gal], api_name=False)
    ck_dd.change(lambda c: I.ckpt_defaults(c, project)["overlap"], inputs=[ck_dd], outputs=[overlap],
                 api_name=False)

    def run(d, progress=gr.Progress()):
        paths = [Path(p) for p in (d[files] or [])]
        if not paths:
            raise gr.Error("Upload at least one photo.")
        if not d[ck_dd]:
            raise gr.Error("Choose an RF-DETR checkpoint (folder with config.json).")
        try:
            per = per_photo_from_table(d[photo_tbl], classes)
            views: list = []
            rows, its, ovs = I.run_inference(
                paths, per, project, ckpt=d[ck_dd], device=device, score=d[score], tile=int(d[tile]),
                overlap=float(d[overlap]), max_side=int(d[max_side] or 0), roi_text=d[roi] or "",
                model_name=d[model_dd], cache=cache,
                normalize=(float(d[n_target]), float(d[n_contrast])) if d[norm_on] else None, views=views,
                corners=d[corners_st] or None, kernel_overlay=bool(d[show_kernel]),
                on_photo=lambda i, n, name: progress((i, n), desc=f"detecting {name}", unit="photos"))
        except (ValueError, FileNotFoundError, OSError) as e:
            raise gr.Error(str(e))
        csv_path = I.write_results_csv(rows, cache / "results.csv")
        notes = sorted({r.get("note") for r in rows if r.get("weight_g") is None and r.get("note")})
        msg = (f"{len(rows)} photo(s) processed with score >= {d[score]}, tile {int(d[tile])}."
               + (" No weight: " + "; ".join(notes) if notes else ""))
        return ovs, _df([[r.get(c) for c in I.RESULT_COLUMNS] for r in rows], I.RESULT_COLUMNS), str(csv_path), \
            msg, its, views

    run_b.click(run, inputs={files, photo_tbl, ck_dd, score, tile, overlap, max_side, roi, model_dd,
                             norm_on, n_target, n_contrast, corners_st, show_kernel},
                outputs=[gallery, res, csv_f, status, items, pre_gal], api_name=False)

    def save(its):
        if not its:
            raise gr.Error("Run inference first.")
        names = I.save_samples(project, its)
        return f"Saved {len(names)} photo(s) as unreviewed samples: {', '.join(names)}. Open the Review tab (reload the photo list)."

    save_b.click(save, inputs=[items], outputs=[status], api_name=False)

    def save_const(rows):
        return I.save_constants(project, rows.values.tolist()), _df(I.constants_rows(project), I.CONST_COLUMNS)

    const_b.click(save_const, inputs=[const_tbl], outputs=[const_msg, const_tbl], api_name=False)


    # ---- scriptable API (used by tests) ------------------------------------------------------
    def api_infer(paths_json: str, ckpt_dir: str, model_name: str = "count_x_constant", score_thr: float = 0.5,
                  crop: str = "", weight_g: float = 0.0, save_samples: bool = False,
                  normalize: bool = True, corners_json: str = "") -> dict:
        """Run inference on photo paths (absolute, or names in W/photos); returns result rows. ``corners_json`` =
        {"<file name>": [[x, y] x 4]} warps those uploads to the A4 canvas first."""
        paths = [Path(p) if Path(p).is_absolute() else project.photos_dir / p for p in json.loads(paths_json)]
        per = {p.name: {"crop": crop or None, "total_weight_g": weight_g or None} for p in paths}
        c = I.ckpt_defaults(ckpt_dir, project)
        rows, its, _ = I.run_inference(paths, per, project, ckpt=ckpt_dir, device=device, score=score_thr,
                                       tile=DEFAULT_TILE, overlap=c["overlap"], max_side=0, roi_text="",
                                       model_name=model_name, cache=cache,
                                       normalize=(TARGET_BRIGHTNESS, DEFAULT_CONTRAST) if normalize else None,
                                       corners=json.loads(corners_json) if corners_json else None)
        saved = I.save_samples(project, its) if save_samples else []
        return {"rows": rows, "saved": saved}

    gr.api(api_infer, api_name="infer")
