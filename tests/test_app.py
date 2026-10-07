import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from detkit.app import logic as L
from detkit.app import proposals as P
from detkit.project import Project
from detkit.schema import IMAGE_FIELDS


@pytest.fixture
def wd(tmp_path):
    p = Project(workdir=tmp_path / "w", classes=["wheat", "sunflower"])
    p.save()
    p.photos_dir.mkdir()
    Image.new("RGB", (4000, 3000), (90, 90, 90)).save(p.photos_dir / "big.jpg")
    Image.new("RGB", (800, 600), (90, 90, 90)).save(p.photos_dir / "small.png")
    return p


def test_form_meta_coercion():
    vals = [None] * len(IMAGE_FIELDS)
    keys = [f.key for f in IMAGE_FIELDS]
    vals[keys.index("total_weight_g")] = "1.25"
    vals[keys.index("manual_count")] = 12.0
    vals[keys.index("pile_id")] = "  p1 "
    meta, errs = L.form_to_meta(vals)
    assert meta == {"total_weight_g": 1.25, "manual_count": 12, "pile_id": "p1"} and not errs
    vals[keys.index("total_weight_g")] = "abc"
    assert L.form_to_meta(vals)[1]
    # clearing a previously set field yields an explicit None
    vals = [None] * len(keys)
    assert L.form_to_meta(vals, {"pile_id": "old"})[0] == {"pile_id": None}
    assert L.meta_to_form({"pile_id": "x"})[keys.index("pile_id")] == "x"


def test_preview_and_scaling(wd):
    prev, size, scale = L.make_preview(wd.photos_dir / "big.jpg", wd.workdir / "c", 1000)
    assert size == (4000, 3000) and scale == 4.0 and Image.open(prev).size == (1000, 750)
    p2, s2, sc2 = L.make_preview(wd.photos_dir / "small.png", wd.workdir / "c", 1000)
    assert sc2 == 1.0 and p2.name == "small.png"


def test_box_roundtrip_and_edit_detection():
    classes = ["wheat", "sunflower"]
    boxes = [{"bbox": [400, 400, 800, 800], "class": "wheat", "origin": "rfdetr", "score": 0.9, "weight_g": 0.04},
             {"bbox": [1200, 1200, 1600, 1600], "class": "wheat", "origin": "vlm", "score": None}]
    ann = L.to_annotator(boxes, 4.0, classes)
    assert ann[0]["xmin"] == 100 and ann[0]["label"] == "wheat"
    same = L.from_annotator(ann, boxes, 4.0, (4000, 3000), classes)
    assert same == boxes                                     # untouched -> identical, extras kept
    ann[0]["xmax"] += 20                                     # resize
    ann[1]["label"] = "sunflower"                            # re-class
    ann.append({"xmin": 500, "ymin": 500, "xmax": 600, "ymax": 600, "label": "wheat"})   # add
    out = L.from_annotator(ann, boxes, 4.0, (4000, 3000), classes)
    assert out[0]["origin"] == "manual" and out[0]["weight_g"] == 0.04 and out[0]["bbox"][2] == 880
    assert out[1]["class"] == "sunflower" and out[1]["origin"] == "vlm"
    assert out[2]["origin"] == "manual" and out[2]["bbox"] == [2000, 2000, 2400, 2400]
    assert len(out) == 3
    del ann[0]                                               # delete
    assert len(L.from_annotator(ann, boxes, 4.0, (4000, 3000), classes)) == 2
    # clamping / degenerate boxes
    bad = [{"xmin": -5, "ymin": 0, "xmax": 2, "ymax": 0.1, "label": "wheat"}]
    assert L.from_annotator(bad, [], 1.0, (100, 100), classes) == []


def test_table_edits_and_stats():
    boxes = [{"bbox": [0, 0, 10, 10], "class": "wheat", "origin": "manual"},
             {"bbox": [0, 0, 20, 10], "class": "sunflower", "origin": "vlm", "score": 0.5}]
    rows = L.boxes_to_rows(boxes)
    rows[0][-2] = "0.04"
    rows[1][-2] = 0.06
    rows[1][-1] = "big"
    out, errs = L.apply_table_edits(boxes, rows)
    assert not errs and out[0]["weight_g"] == 0.04 and out[1]["box_notes"] == "big"
    rows[0][-2] = "x"
    assert L.apply_table_edits(boxes, rows)[1]
    st = L.compute_stats(out, {"scale_mm_per_px": 0.5, "total_weight_g": 0.1}, ["wheat", "sunflower"])
    assert st["n"] == 2 and st["total_area_px"] == 300 and st["total_area_mm2"] == 75
    assert st["sum_box_weight_g"] == pytest.approx(0.1) and "OK" in st["weight_hint"]
    assert "MORE" in L.weight_hint(2, 0.05, 2, 0.1) and "per grain" in L.weight_hint(4, 1.0, 0, None)
    assert "differs" in L.weight_hint(3, None, 0, None, manual_count=5)


def test_delete_boxes():
    boxes = [{"bbox": [0, 0, i + 1, i + 1], "class": "wheat"} for i in range(4)]
    left, gone = L.delete_boxes(boxes, "2, 4")
    assert gone == [2, 4] and [b["bbox"][2] for b in left] == [1, 3]
    assert L.delete_boxes(boxes, "1")[0] == boxes[1:]
    for bad in ("", "5", "0", "x"):
        with pytest.raises(ValueError):
            L.delete_boxes(boxes, bad)


def test_add_class_appends_and_persists(wd):
    L.add_class(wd, " corn ")
    q = Project.load(wd.workdir)
    assert q.classes == ["wheat", "sunflower", "corn"]
    for bad in ("", "Wheat", "a,b"):
        with pytest.raises(ValueError):
            L.add_class(wd, bad)


def test_add_image_field_persists_and_saves(wd):
    f = L.add_image_field(wd, " Moisture pct ", "float", "%")
    assert (f.key, f.kind, f.unit) == ("moisture_pct", "float", "%")
    L.add_image_field(wd, "annotator")
    q = Project.load(wd.workdir)
    assert [x.key for x in q.image_fields()][-2:] == ["moisture_pct", "annotator"]
    for bad in ("", "crop", "reviewed", "annotator", "1st", "a-b"):
        with pytest.raises(ValueError):
            L.add_image_field(q, bad)
    with pytest.raises(ValueError):
        L.add_image_field(q, "x", "date")
    with pytest.raises(ValueError):
        L.add_image_field(q, "x", max_extra=2)
    L.save_review(q, "small.png", [], {"moisture_pct": "12.5", "annotator": " ana "})
    meta = q.store().get_image("small.png")[1]
    assert meta["moisture_pct"] == 12.5 and meta["annotator"] == "ana"
    assert L.meta_to_form(meta, q.image_fields())[-2:] == [12.5, "ana"]
    with pytest.raises(ValueError):
        L.save_review(q, "small.png", [], {"moisture_pct": "wet"})


def test_import_photos_no_overwrite(wd, tmp_path):
    src = tmp_path / "small.png"
    Image.new("RGB", (10, 10)).save(src)
    (tmp_path / "x.txt").write_text("no")
    assert L.import_photos([src, tmp_path / "x.txt"], wd.photos_dir) == ["small_1.png"]
    assert L.list_photos(wd.photos_dir) == ["big.jpg", "small.png", "small_1.png"]


def test_combine_modes():
    ex = [{"origin": "manual", "bbox": [0, 0, 1, 1]}, {"origin": "vlm", "bbox": [1, 1, 2, 2]}]
    new = [{"origin": "rfdetr", "bbox": [3, 3, 4, 4]}]
    assert [b["origin"] for b in L.combine(ex, new, L.MODES[0])] == ["manual", "rfdetr"]
    assert len(L.combine(ex, new, "replace all")) == 1 and len(L.combine(ex, new, "append")) == 3


def test_label_mapping_and_dets():
    cl = ["wheat", "sunflower"]
    assert P.map_label("Wheat", cl, None, False) == "wheat"
    assert P.map_label("seed", cl, "sunflower", True) == "sunflower"
    assert P.map_label("seed", cl, "sunflower", False) is None
    dets = [{"bbox": [10, 20, 30, 40], "label": "seed", "score": 0.8}, {"bbox": [0, 0, 5, 5], "label": "??", "score": 0.5}]
    boxes, dropped = P.dets_to_boxes(dets, cl, "wheat", origin="rfdetr", single_class_model=False, offset=(100, 200), factor=0.5)
    assert boxes == [] and dropped == 2
    boxes, dropped = P.dets_to_boxes(dets, cl, "wheat", origin="rfdetr", single_class_model=True, offset=(100, 200), factor=0.5)
    assert boxes[0]["bbox"] == [120, 240, 160, 280] and boxes[0]["class"] == "wheat" and dropped == 0


def test_propose_rfdetr_fake_predictor():
    seen = {}

    def fake_predict(im, model, proc, device, *, tile, overlap, score):
        seen.update(size=im.size, tile=tile, score=score)
        return [{"bbox": [10, 10, 50, 50], "score": 0.9, "label": "seed", "label_id": 0}]

    model = SimpleNamespace(config=SimpleNamespace(id2label={0: "seed"}))
    bundle = (model, None, {}, "cpu")
    img = Image.new("RGB", (4000, 2000))
    boxes, info = P.propose_rfdetr(img, bundle, ["wheat", "sunflower"], "sunflower", score=0.4, tile=800,
                                   max_side=1000, roi=(1000, 500, 3000, 1500), predict_fn=fake_predict)
    assert seen["size"] == (1000, 500) and seen["score"] == 0.4          # ROI 2000x1000 -> factor 0.5
    assert boxes[0]["bbox"] == [1020, 520, 1100, 600] and boxes[0]["class"] == "sunflower" \
        and boxes[0]["origin"] == "rfdetr"
    with pytest.raises(ValueError):                                       # single-class model, unknown crop
        P.propose_rfdetr(img, bundle, ["wheat", "sunflower"], None, predict_fn=fake_predict)
    assert P.parse_roi("", (10, 10)) is None and P.parse_roi("0,0,99999,50", (100, 80)) == (0, 0, 100, 50)


def test_find_checkpoints(tmp_path):
    for n in ("checkpoints/r1/final", "weights/base/final"):
        d = tmp_path / n
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
    got = P.find_checkpoints(tmp_path)
    assert str(tmp_path / "checkpoints/r1/final") in got and str(tmp_path / "weights/base/final") in got


def test_save_review_store_roundtrip(wd):
    r = L.save_review(wd, "big.jpg", [
        {"bbox": [100, 100, 200, 220], "class": "wheat", "origin": "rfdetr", "score": 0.7, "weight_g": 0.05},
        {"bbox": [-50, -50, 0.2, 0.2], "class": "wheat"}],              # degenerate -> dropped
        {"crop": "wheat", "total_weight_g": "1.5"}, proposer="rfdetr")
    assert r["boxes"] == 1 and r["size"] == [4000, 3000]
    boxes, meta = wd.store().get_image("big.jpg")
    assert boxes[0]["bbox"] == [100, 100, 200, 220] and boxes[0]["weight_g"] == 0.05
    assert meta["reviewed"] and meta["total_weight_g"] == 1.5 and meta["proposer"] == "rfdetr"
    with pytest.raises(ValueError):
        L.save_review(wd, "big.jpg", [{"bbox": [1, 1, 5, 5], "class": "corn"}], {})
    rows = L.photo_rows(L.list_photos(wd.photos_dir), wd.store())
    assert rows[0][:3] == ["big.jpg", "reviewed", 1] and rows[1][1] == "todo"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_app_starts_and_saves_via_client(wd):
    pytest.importorskip("gradio")
    client_mod = pytest.importorskip("gradio_client")
    from detkit.app.ui import build_app

    demo = build_app(wd.workdir)
    port = _free_port()
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True,
                allowed_paths=[str(wd.workdir)])
    try:
        c = client_mod.Client(f"http://127.0.0.1:{port}", verbose=False)
        stt = c.predict(api_name="/status")
        assert stt["photos"] == ["big.jpg", "small.png"] and stt["classes"] == ["wheat", "sunflower"]
        boxes = json.dumps([{"bbox": [10, 10, 90, 70], "class": "sunflower", "weight_g": 0.03}])
        r = c.predict("small.png", boxes, json.dumps({"crop": "sunflower", "total_weight_g": 0.5}), api_name="/save_review")
        assert r["boxes"] == 1
        assert c.predict(api_name="/status")["reviewed"] == ["small.png"]
        b, m = wd.store().get_image("small.png")
        assert b[0]["class"] == "sunflower" and b[0]["origin"] == "manual" and m["total_weight_g"] == 0.5
    finally:
        demo.close()


# ---- inference tab -----------------------------------------------------------------------------
from detkit import weight as W                      # noqa: E402
from detkit.app import inference as I               # noqa: E402


def _fake_propose(monkeypatch, n=4):
    """Replace checkpoint loading + RF-DETR with a deterministic fake (no torch, no GPU)."""
    monkeypatch.setattr(P, "load_bundle", lambda ckpt, device="auto": ("bundle", ckpt, device))

    def fake(image, bundle, classes, crop, **kw):
        cls = crop or classes[0]
        return ([{"bbox": [10 + 60 * i, 10, 50 + 60 * i, 60], "class": cls, "origin": "rfdetr", "score": 0.9}
                 for i in range(n)], {})

    monkeypatch.setattr(P, "propose_rfdetr", fake)


def test_default_score_and_ckpt_defaults(wd, tmp_path):
    assert I.default_score(wd.workdir)[0] == 0.5
    rep = wd.workdir / "reports" / "eval_x"
    rep.mkdir(parents=True)
    (rep / "eval.json").write_text(json.dumps({"best_f1": {"score": 0.1873}}))
    assert I.default_score(wd.workdir)[0] == 0.187
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "detkit_meta.json").write_text(json.dumps({"tile_params": {"tile": 640, "overlap": 0.1}}))
    assert I.ckpt_defaults(str(ck), wd) == {"tile": 640, "overlap": 0.1}
    assert I.ckpt_defaults(None, wd)["tile"] == wd.tile


def test_estimate_row_baseline_todo_and_error():
    dets = [{"bbox": [0, 0, 10, 10], "score": 0.9, "label": "wheat"}] * 5
    cfg = W.load_config("/nonexistent")
    row = I.estimate_row("a.jpg", dets, {"crop": "wheat", "total_weight_g": 0.25}, W.CountTimesConstant(cfg["constants"]), cfg)
    assert row["count"] == 5 and row["weight_g"] == 0.2 and row["error_pct"] == -20.0 and row["counts"] == "wheat 5"
    assert "cost" not in row
    uncal = I.estimate_row("a.jpg", dets, {"crop": "wheat"}, I.load_weight_model("/x", "area_x_constant", cfg), cfg)
    assert uncal["weight_g"] is None and "calibrate" in uncal["note"] and uncal["count"] == 5
    assert [n for _, n in I.model_choices()] == ["count_x_constant", "area_x_constant"]


def test_results_csv():
    rows = [{"file": "a.jpg", "counts": "wheat 2", "count": 2, "weight_g": 0.08, "note": ""}]
    text = I.results_csv(rows)
    assert text.splitlines()[0].startswith("file,counts,count") and "a.jpg,wheat 2,2" in text.splitlines()[1]


def test_run_inference_and_save_samples_roundtrip(wd, tmp_path, monkeypatch):
    _fake_propose(monkeypatch, n=4)
    src = tmp_path / "up"
    src.mkdir()
    for n in ("small.png", "new.png"):
        Image.new("RGB", (800, 600), (80, 80, 80)).save(src / n)
    rows, items, ovs = I.run_inference(
        [src / "small.png", src / "new.png"], {"small.png": {"crop": "wheat", "total_weight_g": 0.2}},
        wd, ckpt="ck", device="cpu", score=0.3, tile=800, overlap=0.2, max_side=0, roi_text="",
        model_name="count_x_constant", cache=tmp_path / "cache")
    assert [r["count"] for r in rows] == [4, 4] and rows[0]["measured_g"] == 0.2 and rows[0]["weight_g"] == 0.16
    assert all(Path(o).exists() for o in ovs) and rows[1]["counts"] == "wheat 4"      # crop defaults to a class
    names = I.save_samples(wd, items)
    assert names == ["small_1.png", "new.png"]                                          # clash -> suffix, no overwrite
    boxes, meta = wd.store().get_image("small_1.png")
    assert len(boxes) == 4 and boxes[0]["origin"] == "rfdetr" and boxes[0]["score"] == 0.9
    assert meta["reviewed"] is False and meta["total_weight_g"] == 0.2
    assert meta["crop"] == "wheat" and (wd.photos_dir / "small.png").exists()


def test_save_constants(wd):
    assert "Saved" in I.save_constants(wd, [["sunflower", 0.07, 2e-6], ["default", "", None]])
    cfg = W.load_config(wd.workdir)
    assert cfg["constants"]["sunflower"] == 0.07 and cfg["area_constants"]["sunflower"] == 2e-6
    assert I.constants_rows(wd)[0][0] == "default" and ["sunflower", 0.07, 2e-6] in I.constants_rows(wd)


def test_infer_via_client(wd, monkeypatch, tmp_path):
    client_mod = pytest.importorskip("gradio_client")
    _fake_propose(monkeypatch, n=3)
    ck = tmp_path / "ck"
    ck.mkdir()
    from detkit.app.ui import build_app

    demo = build_app(wd.workdir)
    port = _free_port()
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True,
                allowed_paths=[str(wd.workdir)])
    try:
        c = client_mod.Client(f"http://127.0.0.1:{port}", verbose=False)
        r = c.predict(json.dumps(["small.png"]), str(ck), "count_x_constant", 0.3, "wheat", 0.1, True, api_name="/infer")
        assert r["rows"][0]["count"] == 3 and r["rows"][0]["weight_g"] == pytest.approx(0.12)
        assert r["saved"] == ["small_1.png"]
        assert wd.store().get_image("small_1.png")[1]["reviewed"] is False
    finally:
        demo.close()


def test_propose_photos_batch(wd, monkeypatch):
    _fake_propose(monkeypatch, n=2)
    L.save_review(wd, "big.jpg", [{"bbox": [0, 0, 9, 9], "class": "wheat"}], {"crop": "wheat"})   # reviewed
    wd.store().upsert_image("small.png", 800, 600, [{"bbox": [700, 500, 750, 550], "class": "sunflower",
                                                       "origin": "manual"}],
                            {"crop": "sunflower", "pile_id": "p1"}, reviewed=False)
    got = list(P.propose_photos(wd, ["big.jpg", "small.png"], "bundle", score=0.5))
    assert got == [("big.jpg", None, "already reviewed"), ("small.png", 3, None)]   # 1 manual kept + 2 new
    boxes, meta = wd.store().get_image("small.png")
    assert [b["origin"] for b in boxes] == ["manual", "rfdetr", "rfdetr"] and boxes[1]["class"] == "sunflower"
    assert meta["reviewed"] is False and meta["pile_id"] == "p1" and meta["proposer"] == "rfdetr"
    assert len(wd.store().get_image("big.jpg")[0]) == 1                                # reviewed untouched
    gen = P.propose_photos(wd, ["small.png"], "bundle", mode="replace all")            # stop after one photo
    next(gen)
    gen.close()
    assert len(wd.store().get_image("small.png")[0]) == 2


def test_batch_propose_ui_handler(wd, monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    _fake_propose(monkeypatch, n=2)
    from detkit.app.ui import build_app

    demo = build_app(wd.workdir)
    fn = next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "run_rf_all")
    by_label = {getattr(b, "label", None): b for b in demo.blocks.values()}
    d = {b: b.value for b in demo.blocks.values() if hasattr(b, "value")}
    d[by_label["RF-DETR checkpoint (folder with config.json)"]] = str(tmp_path)
    d[by_label["Boxes"]] = __import__("pandas").DataFrame(columns=L.table_headers())     # empty box table
    state = next(b for b in demo.blocks.values() if type(b).__name__ == "State")
    d[state] = {"files": [], "file": None, "size": None, "scale": 1.0, "boxes": [], "sig0": "[]", "form0": [],
                "proposer": None, "drafts": {}}
    outs = list(fn(d, progress=lambda *a, **k: None))
    assert len(outs) == 3 and "Batch done: 2/2 photos, 4 boxes" in outs[-1][4]
    assert all(len(wd.store().get_image(f)[0]) == 2 for f in ("big.jpg", "small.png"))


def _ui(wd):
    """The app's event handlers by name + the components of the Review form (no server)."""
    from detkit.app.ui import build_app

    demo = build_app(wd.workdir)
    fns = {getattr(f.fn, "__name__", ""): f for f in demo.fns.values()}
    out_blocks = fns["start"].outputs                 # OUT = [ann, st, table, stats, status, plist, counter, *comps]
    return demo, {k: f.fn for k, f in fns.items()}, out_blocks


def _form_input(out_blocks, out, s, **values):
    pd = pytest.importorskip("pandas")
    d = {b: None for b in out_blocks}
    d[out_blocks[1]], d[out_blocks[2]] = s, pd.DataFrame(columns=L.table_headers())
    comps = out_blocks[7:]
    for b, v in zip(comps, out[7:]):
        d[b] = v["value"] if isinstance(v, dict) else v
    keys = [f.key for f in IMAGE_FIELDS] + [e["key"] for e in Project.load(s["wd"]).extra_image_fields] \
        if "wd" in s else None
    for k, v in values.items():
        d[comps[keys.index(k)]] = v
    return d


def test_extra_property_after_refresh_and_stale_page(wd):
    pytest.importorskip("gradio")
    _demo, fn, ob = _ui(wd)                                # page built before any extra property exists
    s = fn["start"](None)[1]
    r = fn["add_prop_ev"](s, "moisture", "number", "%")
    assert r[0]["n_extra"] == 1 and r[1]["visible"] is True
    r = fn["add_prop_ev"](r[0], "  ", "text", "")         # empty name: a message, no exception
    assert "Type a name" in r[-3]
    out = fn["start"](None)                                # browser refresh: same (stale) layout
    slot0 = out[7 + len(IMAGE_FIELDS)]
    assert out[1]["n_extra"] == 1 and slot0["visible"] is True and slot0["label"] == "moisture (%)"
    r = fn["add_prop_ev"](out[1], "moisture", "number", "")   # again: message + box shown, no exception
    assert "already exists" in r[-3] and r[-2] == "moisture"
    L.save_review(Project.load(wd.workdir), "big.jpg", [], {"crop": "wheat", "moisture": 11})
    # a page that does not show the moisture box yet (e.g. other tab) saves: the stored value is kept
    s = dict(out[1], n_extra=0, wd=wd.workdir)
    fn["save_only"](_form_input(ob, out, s, crop="wheat", pile_id="p1"))
    meta = wd.store().get_image("big.jpg")[1]
    assert meta["moisture"] == 11 and meta["pile_id"] == "p1"


def test_unsaved_edits_survive_refresh(wd):
    pytest.importorskip("gradio")
    _demo, fn, ob = _ui(wd)
    out = fn["start"](None)
    s = dict(out[1], wd=wd.workdir)
    fn["on_form"](_form_input(ob, out, s, pile_id="p9"))         # typed, not saved
    drafts = json.loads((wd.workdir / ".app_cache" / "drafts.json").read_text())
    assert drafts["big.jpg"]["form"][[f.key for f in IMAGE_FIELDS].index("pile_id")] == "p9"
    _demo2, fn2, ob2 = _ui(wd)                                    # refresh / app restart
    out2 = fn2["start"](None)
    assert "Restored your unsaved edits" in out2[4] and "p9" in out2[7:]
    fn2["save_only"](_form_input(ob2, out2, dict(out2[1], wd=wd.workdir)))
    assert wd.store().get_image("big.jpg")[1]["pile_id"] == "p9"
    assert "big.jpg" not in json.loads((wd.workdir / ".app_cache" / "drafts.json").read_text())
    out3 = _ui(wd)[1]["start"](None)                               # opening a photo does not create a draft
    assert "Restored" not in out3[4]
    assert json.loads((wd.workdir / ".app_cache" / "drafts.json").read_text()) == {}
