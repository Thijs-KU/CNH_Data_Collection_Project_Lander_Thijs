"""Gradio layout + events of the "Inference" tab (logic lives in inference.py)."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

from ..preprocess import DEFAULT_CONTRAST, TARGET_BRIGHTNESS
from ..project import DEFAULT_SCORE, DEFAULT_TILE
from . import inference as I
from . import proposals as P

PHOTO_COLS = ["file", "crop", "total_weight_g"]

TODO_TEXT = (
    "**Student tasks** - `low_g` / `high_g` (the bounds of each estimate) stay empty until task 3. "
    "Calibrate the constants on your own weighed photos first (`detkit weight calibrate`, task 2). "
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
                roi = gr.Textbox(label="ROI x0,y0,x1,y1 (original px, optional)")
            norm_on = gr.Checkbox(True, label="Normalize brightness / contrast first (same as the Preprocessing tab)",
                                  info="uploads are made as bright as the project photos; photos already in the "
                                       "project are not changed again")
            with gr.Row():
                n_target = gr.Slider(60, 240, TARGET_BRIGHTNESS, step=1, label="target brightness (median, 0-255)")
                n_contrast = gr.Slider(0, 1, DEFAULT_CONTRAST, step=0.05, label="contrast enhancement (0 = off)")
            run_b = gr.Button("Run inference", variant="primary")
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

    def preview(paths, on, tgt, con):
        """Show the normalization before anything is detected (no model needed)."""
        ps = [Path(p) for p in (paths or [])]
        if not ps or not on:
            return []
        try:
            return I.prepare_photos(ps, project, cache, (float(tgt), float(con)))[1]
        except (ValueError, OSError) as e:
            raise gr.Error(str(e))

    for trig in (files.change, norm_on.change, n_target.release, n_contrast.release):
        trig(preview, inputs=[files, norm_on, n_target, n_contrast], outputs=[pre_gal], api_name=False)
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
                             norm_on, n_target, n_contrast},
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
                  normalize: bool = True) -> dict:
        """Run inference on photo paths (absolute, or names in W/photos); returns result rows."""
        paths = [Path(p) if Path(p).is_absolute() else project.photos_dir / p for p in json.loads(paths_json)]
        per = {p.name: {"crop": crop or None, "total_weight_g": weight_g or None} for p in paths}
        c = I.ckpt_defaults(ckpt_dir, project)
        rows, its, _ = I.run_inference(paths, per, project, ckpt=ckpt_dir, device=device, score=score_thr,
                                       tile=DEFAULT_TILE, overlap=c["overlap"], max_side=0, roi_text="",
                                       model_name=model_name, cache=cache,
                                       normalize=(TARGET_BRIGHTNESS, DEFAULT_CONTRAST) if normalize else None)
        saved = I.save_samples(project, its) if save_samples else []
        return {"rows": rows, "saved": saved}

    gr.api(api_infer, api_name="infer")
