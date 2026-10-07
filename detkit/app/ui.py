"""Gradio UI. ``build_app(workdir, ...)`` returns a ``gr.Blocks``; ``launch(...)`` serves it.

Box editing uses the optional ``gradio_image_annotation`` component (add / move / resize / delete /
re-class by mouse). Without it the app falls back to stock ``gr.AnnotatedImage`` (display only) plus
an editable box table where boxes can be added, edited and deleted by coordinates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import pandas as pd

from ..project import Project
from ..schema import BOX_FIELDS, IMAGE_FIELDS
from . import logic as L
from . import proposals as P
from .inference_ui import build_inference_tab

try:                                                        # optional component
    from gradio_image_annotation import image_annotator
except Exception:                                           # pragma: no cover - depends on install
    image_annotator = None

HAVE_EDITOR = image_annotator is not None
CACHE = ".app_cache"
DRAFTS = "drafts.json"                                      # unsaved edits per photo (survive a refresh / restart)
MAX_EXTRA = 12                                              # hidden form slots for extra photo properties
KIND_NAMES = {"text": "str", "number": "float", "whole number": "int"}
ANN_LABEL = ("Boxes: pick the box tool (first icon) and drag = new box; drag corners/edges = resize; drag inside = "
             "move; select + Delete key = remove; label tool / double-click = change class")


def _num(f):
    return gr.Number(label=L.field_label(f), info=f.help or None, value=None,
                     precision=0 if f.kind == "int" else None)


def make_field_component(f, classes: list[str]):
    """One form component per schema Field (this is what makes the form schema-driven)."""
    if f.key == "crop":
        return gr.Dropdown(choices=classes, value=None, label="crop / class", info=f.help,
                           allow_custom_value=False)
    if f.kind in ("float", "int"):
        return _num(f)
    if f.kind == "choice":
        return gr.Dropdown(choices=list(f.choices), value=None, label=f.key, info=f.help or None,
                           allow_custom_value=True)
    return gr.Textbox(label=f.key, info=f.help or None, lines=1)


def extra_slot_props(project, i: int) -> dict:
    """Label/visibility of extra-property slot *i* (a Textbox; values are typed on save)."""
    extra = project.image_fields()[len(IMAGE_FIELDS):]
    if i >= len(extra):
        return {"visible": False}
    f = extra[i]
    kind = next(k for k, v in KIND_NAMES.items() if v == f.kind)
    return {"visible": True, "label": L.field_label(f), "info": f"extra property ({kind})"}


def _df(rows, headers):
    return pd.DataFrame(rows, columns=headers)


def build_app(workdir: str | Path, *, ckpt: str | None = None, device: str | None = None) -> gr.Blocks:
    project = Project.load(workdir)
    wd = Path(workdir)
    cache = wd / CACHE
    photos = project.photos_dir
    photos.mkdir(parents=True, exist_ok=True)
    dev = device or project.device

    def ckpts() -> list[str]:
        return P.find_checkpoints(wd, ckpt, project.proposer_ckpt)

    def load_drafts() -> dict:
        try:
            return json.loads((cache / DRAFTS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def write_drafts(drafts: dict) -> None:
        cache.mkdir(parents=True, exist_ok=True)
        tmp = cache / (DRAFTS + ".tmp")
        tmp.write_text(json.dumps(drafts, default=str), encoding="utf-8")
        tmp.replace(cache / DRAFTS)                         # atomic: a crash never leaves half a file

    def new_state() -> dict:
        # n_extra: extra-property boxes this page shows (a save never touches the ones it does not show)
        return {"files": L.list_photos(photos), "file": None, "size": None, "scale": 1.0,
                "boxes": [], "sig0": "[]", "form0": [], "proposer": None, "drafts": load_drafts(),
                "n_extra": len(project.extra_image_fields)}

    # ---------------------------------------------------------------- components
    with gr.Blocks(title=f"detkit annotation - {wd.name}") as demo:
        gr.Markdown(f"## detkit annotation app - project `{wd.name}`\n"
                    "Review the proposed boxes (add / move / resize / delete / change class), fill in the "
                    "metadata, then **Save & next**. Saved photos feed `detkit split --from-reviewed`.")
        st = gr.State(new_state())
        with gr.Tabs():
            with gr.Tab("Review"):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            prev_b = gr.Button("< Prev")
                            next_b = gr.Button("Next >")
                            nextu_b = gr.Button("Next unreviewed >>")
                            counter = gr.Markdown("")
                        if HAVE_EDITOR:
                            ann = image_annotator(
                                None, label_list=list(project.classes), label_colors=L.class_colors(project.classes),
                                image_type="filepath", sources=[], height=640, use_default_label=True,
                                show_download_button=False, show_clear_button=False, show_remove_button=False,
                                label=ANN_LABEL)
                        else:
                            ann = gr.AnnotatedImage(label="Boxes (display only: install gradio_image_annotation "
                                                          "to edit on the image)", height=640)
                        gr.Markdown("Box table (weights / notes are editable here; "
                                    + ("boxes are drawn on the image above)." if HAVE_EDITOR else
                                       "add / delete rows or edit coordinates to edit boxes)."))
                        box_headers = L.table_headers() + ([] if HAVE_EDITOR else ["x1", "y1", "x2", "y2"])
                        table = gr.Dataframe(
                            headers=box_headers, value=_df([], box_headers), interactive=True, wrap=True,
                            row_count=(0, "dynamic" if not HAVE_EDITOR else "fixed"), max_height=260,
                            static_columns=list(range(len(L.table_headers()) - len(BOX_FIELDS))) if HAVE_EDITOR else None,
                            label="Boxes")
                        with gr.Row():
                            del_num = gr.Textbox(label="box # to delete (click a row in the table, or type e.g. 3 or 2,5)",
                                                 scale=3)
                            del_b = gr.Button("Delete box", variant="stop", scale=1)
                        with gr.Accordion("Proposals (get starting boxes)", open=False):
                            ck = ckpts()
                            with gr.Row():
                                ck_dd = gr.Dropdown(choices=ck, value=ck[0] if ck else None, allow_custom_value=True,
                                                    label="RF-DETR checkpoint (folder with config.json)")
                                mode = gr.Radio(list(L.MODES), value=L.MODES[0], label="When proposing")
                            with gr.Row():
                                score = gr.Slider(0.05, 0.95, 0.5, step=0.01, label="score threshold")
                                tile = gr.Slider(200, 1600, project.tile, step=50, label="tile (px)")
                                overlap = gr.Slider(0.0, 0.5, project.overlap, step=0.05, label="overlap")
                                max_side = gr.Slider(0, 6000, 0, step=100,
                                                     label="downscale longest side to (0 = off; faster on CPU)")
                            roi = gr.Textbox(label="ROI x0,y0,x1,y1 in original pixels (optional, empty = whole photo)")
                            with gr.Row():
                                rf_b = gr.Button("Propose with RF-DETR", variant="primary")
                                clr_b = gr.Button("Start empty (remove all boxes)")
                            with gr.Row():
                                all_b = gr.Button("Propose for all unreviewed photos", variant="secondary")
                                stop_b = gr.Button("Stop", variant="stop")
                            gr.Markdown("Batch: runs the checkpoint, score threshold, tile / overlap / downscale and "
                                        "*When proposing* above on every unreviewed photo (each photo's own crop) and "
                                        "stores the boxes unreviewed. Reviewed photos and photos with unsaved edits "
                                        "are skipped. *Stop* keeps the photos already done.")
                            if not ck:
                                gr.Markdown("No checkpoint found (looked in `checkpoints/*/final`, `weights/`). "
                                            "Type a path or draw boxes by hand.")
                    with gr.Column(scale=2):
                        gr.Markdown("### Photo metadata")
                        gr.Markdown("Grain moisture is assumed constant for all samples and is not recorded "
                                    "(add it below as an extra property if you measure it).")
                        comps = [make_field_component(f, project.classes) for f in IMAGE_FIELDS]
                        crop_dd = comps[[f.key for f in IMAGE_FIELDS].index("crop")]
                        slots = [gr.Textbox(lines=1, **extra_slot_props(project, i)) for i in range(MAX_EXTRA)]
                        comps += slots
                        with gr.Accordion("Add an extra property", open=False):
                            gr.Markdown("A new box in this form for every photo (e.g. annotator, moisture). "
                                        "Saved in `project.json`, so it is there the next time the app opens.")
                            with gr.Row():
                                np_name = gr.Textbox(label="name (e.g. annotator, moisture_pct)", scale=2)
                                np_kind = gr.Dropdown(list(KIND_NAMES), value="text", label="type", scale=1)
                                np_unit = gr.Textbox(label="unit (optional, e.g. %)", scale=1)
                            np_b = gr.Button("Add an extra property")
                        stats = gr.Markdown("")
                        with gr.Row():
                            save_b = gr.Button("Save", variant="secondary")
                            savenext_b = gr.Button("Save & next", variant="primary")
                        status = gr.Markdown("")
                        with gr.Accordion("Photos", open=True):
                            plist = gr.Dataframe(headers=["file", "status", "boxes", "crop"], interactive=False,
                                                 max_height=280, label="click a row to open")
                            up = gr.File(file_count="multiple", file_types=["image"], label="Add photos",
                                         type="filepath")
                        with gr.Accordion("Add a new crop / class", open=False):
                            nc_name = gr.Textbox(label="name (e.g. corn)")
                            nc_b = gr.Button("Add class")
                            gr.Markdown("Classes are only ever appended (ids never change). Retrain after "
                                        "labelling a few photos of the new crop.")

            with gr.Tab("Inference"):
                build_inference_tab(project, wd, ckpt, dev)

        # ---------------------------------------------------------------- helpers
        form_inputs = set(comps)

        def form_of(d) -> list:
            n = len(IMAGE_FIELDS) + min(d[st].get("n_extra", 0), MAX_EXTRA)
            return [d[c] for c in comps[:n]]

        def pad(form) -> list:
            """Form values for all components (unused extra slots -> None)."""
            return (list(form or []) + [None] * len(comps))[:len(comps)]

        def cur_crop(d) -> str | None:
            return d[crop_dd] or None

        def sync_boxes(d) -> tuple[dict, list[str]]:
            """Apply table edits (weights/notes) to the state boxes."""
            s = dict(d[st])
            if HAVE_EDITOR:
                s["boxes"], errs = L.apply_table_edits(s["boxes"], d[table].values.tolist())
            else:
                s["boxes"], errs = _boxes_from_full_table(s, d[table])
            return s, errs

        def _boxes_from_full_table(s, df) -> tuple[list[dict], list[str]]:
            """Fallback editor: the table holds all box data (class, x1..y2, extras)."""
            errs, out, prev = [], [], s["boxes"]
            hd = list(df.columns)
            for i, r in enumerate(df.to_dict(orient="records")):
                try:
                    bb = L.clamp_box([r["x1"], r["y1"], r["x2"], r["y2"]], s["size"])
                except (TypeError, ValueError):
                    bb = None
                if bb is None or r.get("class") not in project.classes:
                    if any(pd.notna(v) and v != "" for v in r.values()):
                        errs.append(f"row {i + 1}: needs valid class and x1,y1,x2,y2 inside the photo")
                    continue
                old = prev[int(r["#"]) - 1] if pd.notna(r.get("#")) and 0 < int(r["#"]) <= len(prev) else None
                b = dict(old) if old else {"origin": "manual", "score": None}
                if old and [round(v, 1) for v in old["bbox"]] != bb:
                    b["origin"], b["score"] = "manual", None
                b.update({"bbox": bb, "class": r["class"]})
                for f in BOX_FIELDS:
                    v = r.get(f.key)
                    b[f.key] = None if (v is None or (isinstance(v, float) and pd.isna(v)) or v == "") else v
                out.append(b)
            return out, errs

        def rows_of(s) -> pd.DataFrame:
            rows = L.boxes_to_rows(s["boxes"])
            if not HAVE_EDITOR:
                rows = [r + [round(b["bbox"][0]), round(b["bbox"][1]), round(b["bbox"][2]), round(b["bbox"][3])]
                        for r, b in zip(rows, s["boxes"])]
            return _df(rows, box_headers)

        def stats_of(s, form) -> str:
            meta, _ = L.form_to_meta(form)
            return L.stats_markdown(L.compute_stats(s["boxes"], meta, project.classes), meta)

        def list_rows(s):
            return _df(L.photo_rows(s["files"], project.store()), ["file", "status", "boxes", "crop"])

        def counter_of(s) -> str:
            meta = project.store().load_meta()
            done = sum(1 for f in s["files"] if meta.get(f, {}).get("reviewed"))
            i = s["files"].index(s["file"]) + 1 if s["file"] in s["files"] else 0
            return f"**{s['file'] or '-'}**  ({i}/{len(s['files'])}, {done} reviewed)"

        def label_props(crop):
            """Editor label list with the photo's crop FIRST: it is the default label of new boxes."""
            order = ([crop] if crop in project.classes else []) + [c for c in project.classes if c != crop]
            return {"label_list": list(order),
                    "label_colors": [L.class_colors(project.classes)[project.classes.index(c)] for c in order]}

        def ann_value(s):
            if s["file"] is None:
                return None
            v = L.to_annotator(s["boxes"], s["scale"], project.classes)
            if HAVE_EDITOR:
                return {"image": s["preview"], "boxes": v}
            return (s["preview"], [((int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"])), b["label"])
                                   for b in v])

        def view(s, form, msg="") -> list:
            av = ann_value(s)
            if HAVE_EDITOR:
                crop = form[[f.key for f in IMAGE_FIELDS].index("crop")] if form else None
                av = gr.update(value=av, label=ANN_LABEL, **label_props(crop))
            s["n_extra"] = len(project.extra_image_fields)             # re-sync the boxes with project.json
            stash(s, form)
            vals, n = pad(form), len(IMAGE_FIELDS)
            slot_out = [gr.update(value=v, **extra_slot_props(project, i)) for i, v in enumerate(vals[n:])]
            return [av, s, rows_of(s), stats_of(s, form), msg, list_rows(s), counter_of(s), *vals[:n], *slot_out]

        OUT = [ann, st, table, stats, status, plist, counter, *comps]

        def edited(s, form) -> bool:
            if json.dumps(s["boxes"], default=str) != s["sig0"]:
                return True
            fields = project.image_fields()[:len(form)]                # compare typed values ("" == None)
            now, errs = L.form_to_meta(list(form), None, fields)
            return bool(errs) or now != L.form_to_meta(list(s["form0"])[:len(form)], None, fields)[0]

        def stash(s, form):
            """Keep unsaved edits of the current photo as a draft (also on disk, so a refresh or an
            app restart does not lose them); drop the draft once the photo matches what is saved."""
            if not s["file"]:
                return
            old = s["drafts"].get(s["file"])
            if edited(s, form):
                s["drafts"][s["file"]] = {"boxes": s["boxes"], "form": list(form), "proposer": s["proposer"]}
            else:
                s["drafts"].pop(s["file"], None)
            if json.dumps(old, default=str) != json.dumps(s["drafts"].get(s["file"]), default=str):
                write_drafts(s["drafts"])

        def open_file(s, name, form_now=None):
            if form_now is not None:
                stash(s, form_now)
            s = dict(s)
            s["files"] = L.list_photos(photos)
            path = photos / name
            preview, size, scale = L.make_preview(path, cache / "previews")
            store = project.store()
            boxes, meta = store.get_image(name)
            saved_form = L.meta_to_form(meta, project.image_fields())
            if len(project.classes) == 1 and not meta.get("crop"):
                saved_form[[f.key for f in IMAGE_FIELDS].index("crop")] = project.classes[0]
            dr = s["drafts"].pop(name, None)
            if dr:
                write_drafts(s["drafts"])                       # view() stores it again if still unsaved
            s.update(file=name, size=size, scale=scale, preview=str(preview), sig0=json.dumps(boxes, default=str),
                     form0=saved_form, proposer=(dr or {}).get("proposer") or meta.get("proposer"))
            if dr:
                s["boxes"], form = dr["boxes"], dr["form"]
            else:
                s["boxes"], form = boxes, saved_form
            msg = ("Restored your unsaved edits for this photo (not saved yet: click Save)." if dr else
                   ("Loaded saved review." if meta.get("reviewed") else
                    ("Loaded unreviewed boxes." if boxes else
                     "No boxes yet: use Proposals or draw boxes.")))
            if max(size) > L.PREVIEW_MAXSIDE:
                msg += f" (Shown downscaled x{scale:.2f}; boxes are stored in original {size[0]}x{size[1]} px.)"
            return view(s, form, msg)

        def do_save(d):
            s, errs = sync_boxes(d)
            form = form_of(d)
            if s["file"] is None:
                raise gr.Error("Open a photo first.")
            if errs:
                raise gr.Error("Fix before saving: " + "; ".join(errs))
            try:
                r = L.save_review(project, s["file"], s["boxes"], form, s["proposer"])
            except ValueError as e:
                raise gr.Error(str(e))
            s["sig0"], s["form0"] = json.dumps(s["boxes"], default=str), form
            s["drafts"].pop(s["file"], None)
            write_drafts(s["drafts"])
            return s, form, r

        # ---------------------------------------------------------------- events
        def sync(d):
            s = dict(d[st])
            if s["file"] is None:
                return s, rows_of(s), ""
            v = d[ann] if HAVE_EDITOR else None
            if HAVE_EDITOR and v is not None:
                s["boxes"] = L.from_annotator(v.get("boxes"), s["boxes"], s["scale"], s["size"],
                                              project.classes, cur_crop(d))
            stash(s, form_of(d))
            return s, rows_of(s), stats_of(s, form_of(d))

        if HAVE_EDITOR:
            ann.change(sync, inputs={ann, st, *comps}, outputs=[st, table, stats], api_name=False,
                       show_progress="hidden")

        def on_table(d):
            s, errs = sync_boxes(d)
            stash(s, form_of(d))
            return s, (rows_of(s) if not HAVE_EDITOR else gr.skip()), stats_of(s, form_of(d)), \
                ("Table: " + "; ".join(errs) if errs else "")

        table.input(on_table, inputs={table, st, *comps}, outputs=[st, table, stats, status], api_name=False,
                    show_progress="hidden")
        if not HAVE_EDITOR:
            table.input(lambda d: (_ann_only(d),), inputs={table, st, *comps}, outputs=[ann], api_name=False)

        def _ann_only(d):
            s, _ = sync_boxes(d)
            return ann_value(s)

        def pick_box(evt: gr.SelectData, df):
            try:
                return str(int(df.iloc[evt.index[0]]["#"]))
            except (TypeError, ValueError, IndexError, KeyError):
                return gr.skip()

        table.select(pick_box, inputs=[table], outputs=[del_num], api_name=False)

        def delete_box(d):
            s, _ = sync_boxes(d)
            if s["file"] is None:
                raise gr.Error("Open a photo first.")
            try:
                s["boxes"], gone = L.delete_boxes(s["boxes"], d[del_num])
            except ValueError as e:
                raise gr.Error(str(e))
            msg = f"Deleted box {', '.join(map(str, gone))}. Not saved yet: click Save."
            return [*view(s, form_of(d), msg), ""]

        del_b.click(delete_box, inputs={st, table, del_num, *comps}, outputs=[*OUT, del_num], api_name=False)

        if HAVE_EDITOR:
            crop_dd.change(lambda c: gr.update(label=ANN_LABEL, **label_props(c)), inputs=[crop_dd], outputs=[ann],
                           api_name=False, show_progress="hidden")

        def on_form(d):
            stash(d[st], form_of(d))
            return stats_of(d[st], form_of(d))

        for c in comps:
            c.input(on_form, inputs={st, *comps}, outputs=[stats], api_name=False, show_progress="hidden")

        def goto(d, target: str):
            s, _ = sync_boxes(d)
            files = L.list_photos(photos)
            if not files:
                return view(s, form_of(d), "No photos yet: upload some.")
            if target == "first":
                name = files[0]
            else:
                i = files.index(s["file"]) if s["file"] in files else -1
                if target == "next":
                    name = files[min(i + 1, len(files) - 1)]
                elif target == "prev":
                    name = files[max(i - 1, 0)]
                else:                                            # next unreviewed
                    meta = project.store().load_meta()
                    todo = [f for f in files[i + 1:] + files[:i + 1] if not meta.get(f, {}).get("reviewed")]
                    if not todo:
                        return view(s, form_of(d), "All photos are reviewed.")
                    name = todo[0]
            return open_file(s, name, form_of(d))

        prev_b.click(lambda d: goto(d, "prev"), inputs={st, table, *comps}, outputs=OUT, api_name=False)
        next_b.click(lambda d: goto(d, "next"), inputs={st, table, *comps}, outputs=OUT, api_name=False)
        nextu_b.click(lambda d: goto(d, "unreviewed"), inputs={st, table, *comps}, outputs=OUT, api_name=False)

        def pick(d, evt: gr.SelectData):
            s, _ = sync_boxes(d)
            row = evt.row_value if evt.row_value else None
            if not row or row[0] == s["file"]:
                return [gr.skip()] * len(OUT)
            return open_file(s, row[0], form_of(d))

        plist.select(pick, inputs={st, table, *comps}, outputs=OUT, api_name=False)

        def save_only(d):
            s, form, r = do_save(d)
            return view(s, form, f"Saved {r['file']}: {r['boxes']} boxes.")

        def save_next(d):
            s, form, r = do_save(d)
            files = L.list_photos(photos)
            i = files.index(s["file"]) if s["file"] in files else -1
            if i + 1 >= len(files):
                return view(s, form, f"Saved {r['file']} ({r['boxes']} boxes). That was the last photo.")
            out = open_file(s, files[i + 1])
            out[4] = f"Saved {r['file']} ({r['boxes']} boxes). " + out[4]
            return out

        save_b.click(save_only, inputs={st, table, *comps}, outputs=OUT, api_name=False)
        savenext_b.click(save_next, inputs={st, table, *comps}, outputs=OUT, api_name=False)

        def uploaded(d):
            s = dict(d[st])
            added = L.import_photos(d[up] or [], photos)
            s["files"] = L.list_photos(photos)
            if not added:
                return s, list_rows(s), counter_of(s), "No image files recognised.", None
            if s["file"] is None:
                out = open_file(s, added[0], form_of(d))
                return out[1], out[5], out[6], f"Added {len(added)} photo(s).", None
            return s, list_rows(s), counter_of(s), f"Added {len(added)} photo(s): {', '.join(added[:5])}", None

        up.upload(uploaded, inputs={up, st, *comps}, outputs=[st, plist, counter, status, up],
                  api_name=False)

        def add_class_ev(name, cur_crop_value):
            try:
                L.add_class(project, name)
            except ValueError as e:
                raise gr.Error(str(e))
            upd = [gr.update(choices=list(project.classes))]
            if HAVE_EDITOR:
                upd.append(gr.update(label=ANN_LABEL, **label_props(cur_crop_value)))
            else:
                upd.append(gr.skip())
            return (*upd, f"Added class '{project.classes[-1]}' (project.json updated).", "")

        nc_b.click(add_class_ev, inputs=[nc_name, crop_dd], outputs=[crop_dd, ann, status, nc_name],
                   api_name=False)

        def add_prop_ev(s, name, kind, unit):
            """Add a property; on any outcome re-sync the boxes with project.json (a box hidden on this page,
            e.g. added in another tab, appears with the open photo's saved value; typed values are kept)."""
            if not (name or "").strip():
                gr.Warning("Type a name for the new property first.")
                msg, keep = "Type a name for the new property first.", False
            else:
                try:
                    f = L.add_image_field(project, name, KIND_NAMES.get(kind, "str"), unit, max_extra=MAX_EXTRA)
                    msg, keep = f"Added property '{f.key}' (project.json updated). Fill it in and Save.", False
                except ValueError as e:
                    gr.Warning(str(e))
                    msg, keep = f"{e}. Its box is shown in the form above.", True
            s = dict(s)
            shown, s["n_extra"] = s.get("n_extra", 0), len(project.extra_image_fields)
            saved = project.store().get_image(s["file"])[1] if s["file"] else {}
            extra = project.image_fields()[len(IMAGE_FIELDS):]
            upd = [gr.skip() if i < shown else
                   gr.update(value=saved.get(extra[i].key) if i < len(extra) else None,
                             **extra_slot_props(project, i)) for i in range(MAX_EXTRA)]
            return (s, *upd, msg, name if keep else "", unit if keep else "")

        np_b.click(add_prop_ev, inputs=[st, np_name, np_kind, np_unit],
                   outputs=[st, *slots, status, np_name, np_unit], api_name=False)

        # proposals -------------------------------------------------------------------------------
        def apply_new(d, new, mode_, proposer, msg):
            s, errs = sync_boxes(d)
            s["boxes"] = L.combine(s["boxes"], new, mode_)
            s["proposer"] = proposer
            return view(s, form_of(d), msg)

        def run_rf(d):
            s, _ = sync_boxes(d)
            if s["file"] is None:
                raise gr.Error("Open a photo first.")
            if not d[ck_dd]:
                raise gr.Error("Choose an RF-DETR checkpoint (folder with config.json), or draw boxes by hand.")
            try:
                from PIL import Image
                bundle = P.load_bundle(d[ck_dd], dev)
                r = P.parse_roi(d[roi], s["size"])
                with Image.open(photos / s["file"]) as im:
                    boxes, info = P.propose_rfdetr(im.convert("RGB"), bundle, project.classes, cur_crop(d),
                                                   score=d[score], tile=d[tile], overlap=d[overlap],
                                                   max_side=int(d[max_side] or 0), roi=r)
            except (ValueError, FileNotFoundError, OSError) as e:
                raise gr.Error(str(e))
            note = f" ({info['dropped_unknown_label']} dropped: label not in project classes)" if info["dropped_unknown_label"] else ""
            return apply_new(d, boxes, d[mode], "rfdetr", f"RF-DETR proposed {len(boxes)} boxes{note}.")

        def run_empty(d):
            return apply_new(d, [], "replace all", "manual", "Cleared all boxes. Draw your own.")

        pin = {st, table, *comps, ck_dd, score, tile, overlap, max_side, roi, mode}
        rf_b.click(run_rf, inputs=pin, outputs=OUT, api_name=False)
        clr_b.click(run_empty, inputs=pin, outputs=OUT, api_name=False)

        def run_rf_all(d, progress=gr.Progress()):
            """Batch proposals (a generator: the status line updates per photo and Stop takes effect)."""
            s, _ = sync_boxes(d)
            stash(s, form_of(d))                                 # unsaved edits of the open photo win
            if not d[ck_dd]:
                raise gr.Error("Choose an RF-DETR checkpoint (folder with config.json).")
            meta = project.store().load_meta()
            files = [f for f in L.list_photos(photos)
                     if not meta.get(f, {}).get("reviewed") and f not in s["drafts"]]
            if not files:
                raise gr.Error("No unreviewed photos without unsaved edits.")
            skip = [gr.skip()] * len(OUT)
            progress((0, len(files)), desc="loading model", unit="photos")
            try:
                bundle = P.load_bundle(d[ck_dd], dev)
            except (ValueError, FileNotFoundError, OSError) as e:
                raise gr.Error(str(e))
            done, n_boxes, failed = 0, 0, []
            batch = P.propose_photos(project, files, bundle, score=d[score], tile=d[tile], overlap=d[overlap],
                                     max_side=int(d[max_side] or 0), mode=d[mode])
            for i, (f, n, why) in enumerate(batch, 1):
                progress((i, len(files)), desc=f"proposed {f}", unit="photos")
                if why:
                    failed.append(f"{f}: {why}")
                else:
                    done, n_boxes = done + 1, n_boxes + n
                out = list(skip)
                out[OUT.index(status)] = f"Batch: {i}/{len(files)} photos ({n_boxes} boxes so far)..."
                out[OUT.index(plist)] = list_rows(s)
                yield out
            msg = (f"Batch done: {done}/{len(files)} photos, {n_boxes} boxes at score >= {d[score]} (unreviewed: "
                   "check and Save each photo).")
            if failed:
                msg += " Skipped: " + "; ".join(failed[:5]) + (" ..." if len(failed) > 5 else "")
            if s["file"]:
                out = open_file(s, s["file"])                    # show the new boxes of the open photo
                out[OUT.index(status)] = msg
                yield out
            else:
                yield view(s, form_of(d), msg)

        all_ev = all_b.click(run_rf_all, inputs=pin, outputs=OUT, api_name=False)
        stop_b.click(lambda: "Batch stopped; photos done so far are stored.", outputs=[status], cancels=[all_ev],
                     api_name=False)

        def start(s):
            s = new_state()
            pending = [f for f in s["files"] if f in s["drafts"]]
            if pending:                                          # e.g. after a refresh: back to the unsaved work
                out = open_file(s, pending[0])
                if len(pending) > 1:
                    out[OUT.index(status)] += (f" {len(pending)} photos have unsaved edits: "
                                               + ", ".join(pending[:5]) + (" ..." if len(pending) > 5 else ""))
                return out
            if s["files"]:
                return open_file(s, s["files"][0])
            return view(s, [None] * len(comps), "No photos yet: upload some below.")

        demo.load(start, inputs=[st], outputs=OUT, api_name=False)

        # scriptable API (also used by the tests) ---------------------------------------------------
        def api_save_review(file_name: str, boxes_json: str, meta_json: str = "{}") -> dict:
            """Save one photo. boxes_json: [{"bbox":[x1,y1,x2,y2],"class":..,"weight_g":..}] in original px."""
            boxes = json.loads(boxes_json)
            for b in boxes:
                b.setdefault("origin", "manual")
            return L.save_review(project, file_name, boxes, json.loads(meta_json), None)

        def api_status() -> dict:
            store = project.store()
            return {"photos": L.list_photos(photos), "reviewed": store.reviewed_files(),
                    "classes": list(project.classes), "editor": HAVE_EDITOR}

        gr.api(api_save_review, api_name="save_review")
        gr.api(api_status, api_name="status")
    return demo


def launch(workdir: str | Path, *, host: str = "127.0.0.1", port: int = 7860, ckpt: str | None = None,
           device: str | None = None, block: bool = True, **kw):
    demo = build_app(workdir, ckpt=ckpt, device=device)
    return demo.launch(server_name=host, server_port=port, allowed_paths=[str(Path(workdir).resolve())],
                       prevent_thread_lock=not block, **kw)
