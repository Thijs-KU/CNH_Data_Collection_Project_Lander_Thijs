import json
import math

import pytest

from detkit.cli import main
from detkit.project import Project
from detkit.weight import (MODELS, AreaTimesConstant, CountTimesConstant, KernelAreaTimesConstant, SampleTable,
                           WeightModel, evaluate_model, get_model, image_features, kernel_mask, load_config,
                           save_config)


def box(x, y, w=10, h=20, label="wheat"):
    return {"bbox": [x, y, x + w, y + h], "score": 0.9, "label": label}


def test_image_features_math():
    f = image_features([box(0, 0, 10, 20), box(5, 5, 20, 30, "sunflower")], {"scale_mm_per_px": 0.5})
    assert f["count"] == 2 and f["count_wheat"] == 1 and f["count_sunflower"] == 1
    assert f["total_area_px"] == 200 + 600 and f["mean_area_px"] == 400 and f["median_area_px"] == 400
    assert f["mean_diag_px"] == pytest.approx((math.hypot(10, 20) + math.hypot(20, 30)) / 2)
    assert f["total_area_mm2"] == pytest.approx(800 * 0.25) and f["mean_area_mm2"] == pytest.approx(100)


def test_image_features_empty_and_no_scale():
    f = image_features([])
    assert f["count"] == 0 and f["total_area_px"] == 0 and f["mean_area_px"] == 0
    assert "total_area_mm2" not in f


def kernel_photo(discs, size=(120, 120), paper=(235, 235, 240), kernel=(40, 160, 230)):
    """White-ish photo (BGR) with filled yellow discs [(cx, cy, r)]."""
    import cv2
    import numpy as np
    img = np.full((size[1], size[0], 3), paper, np.uint8)
    for cx, cy, r in discs:
        cv2.circle(img, (cx, cy), r, kernel, -1)
    return img


def test_kernel_features_count_only_kernel_pixels():
    img = kernel_photo([(30, 30, 10), (80, 70, 14)])
    dets = [{"bbox": [10, 10, 50, 50], "score": 0.9, "label": "wheat"},       # 40x40 box around a r=10 disc
            {"bbox": [60, 50, 100, 90], "score": 0.9, "label": "wheat"}]      # 40x40 box around a r=14 disc
    f = image_features(dets, {"scale_mm_per_px": 0.5}, img)
    truth = math.pi * (10 ** 2 + 14 ** 2)
    assert f["total_area_px"] == 3200
    assert f["kernel_area_px"] == pytest.approx(truth, rel=0.08)              # ~ the discs, not the boxes
    assert f["kernel_area_px"] < 0.35 * f["total_area_px"]
    assert f["kernel_fill_ratio"] == pytest.approx(f["kernel_area_px"] / 3200)
    assert f["mean_kernel_area_px"] == pytest.approx(f["kernel_area_px"] / 2, rel=0.05)
    assert f["kernel_area_mm2"] == pytest.approx(f["kernel_area_px"] * 0.25)
    assert "kernel_area_px" not in image_features(dets)                        # no image -> old behaviour


def test_kernel_mask_overlapping_boxes_count_once_and_edge_cases():
    img = kernel_photo([(50, 50, 12)])
    box_a, box_b = [30, 30, 70, 70], [40, 40, 80, 80]                          # overlap on the same kernel
    one, _ = kernel_mask(img, [box_a])
    both, areas = kernel_mask(img, [box_a, box_b])
    assert both.sum() == pytest.approx(one.sum(), rel=0.05) and len(areas) == 2   # counted once, not twice
    assert kernel_mask(img, [])[0].sum() == 0
    assert kernel_mask(img, [[5, 5, 5, 20]])[1] == [0]                         # empty box
    assert kernel_mask(kernel_photo([]), [[10, 10, 50, 50]])[0].sum() == 0    # empty paper


def test_kernel_area_model_and_table(tmp_path):
    p = make_project(tmp_path)
    import cv2
    p.photos_dir.mkdir(parents=True, exist_ok=True)
    for rec in p.store().load_coco()["images"]:
        cv2.imwrite(str(p.photos_dir / rec["file_name"]), kernel_photo([(5, 10, 4)], size=(200, 40)))
    t = SampleTable.from_project(p).with_weight()
    assert all("kernel_area_px" in s.features for s in t)
    m = KernelAreaTimesConstant().fit(t)
    s = t.rows[0]
    assert m.predict(s.features, s.crop).weight_g == pytest.approx(s.features["kernel_area_px"] * m.constants[s.crop])
    assert all("kernel_area_px" not in s.features for s in SampleTable.from_project(p, kernel_area=False))


def make_project(tmp_path, n_per=(2, 4, 6, 8)):
    p = Project(workdir=tmp_path, classes=["wheat", "sunflower"])
    p.save()
    st = p.store()
    for i, n in enumerate(n_per):
        st.upsert_image(f"w{i}.jpg", 1000, 1000, [{"bbox": [k * 20, 0, k * 20 + 10, 20], "class": "wheat"}
                                                  for k in range(n)],
                        {"crop": "wheat", "total_weight_g": 0.05 * n + (0.01 if i % 2 else 0)},
                        reviewed=bool(i % 2))
    st.upsert_image("s0.jpg", 1000, 1000, [{"bbox": [0, 0, 10, 10], "class": "sunflower"}], {"crop": "sunflower"})
    return p


def test_sample_table_from_project(tmp_path):
    t = SampleTable.from_project(make_project(tmp_path))
    assert len(t) == 5 and len(t.with_weight()) == 4 and len(t.for_crop("sunflower")) == 1
    s = {r.file: r for r in t}["w1.jpg"]
    assert s.features["count"] == 4 and s.features["count_wheat"] == 4 and s.crop == "wheat"
    assert s.weight_g == pytest.approx(0.21)
    assert len(SampleTable.from_project(make_project(tmp_path), reviewed_only=True)) == 3   # w1, w3, s0
    rec = t.to_records()[0]
    assert {"file", "crop", "weight_g", "count"} <= set(rec)
    t.to_csv(tmp_path / "o.csv")
    assert (tmp_path / "o.csv").read_text().startswith("file,crop,weight_g")


def test_sample_table_from_predictions(tmp_path):
    p = make_project(tmp_path)
    pj = tmp_path / "pred.json"
    pj.write_text(json.dumps({"/x/w0.jpg": {"detections": [box(0, 0), box(30, 0), box(60, 0)]}}))
    t = SampleTable.from_project(p, use="predictions", predictions_path=pj)
    assert len(t) == 1 and t.rows[0].features["count"] == 3 and t.rows[0].weight_g == pytest.approx(0.1)


def test_fit_stores_spread_per_pile(tmp_path):
    import statistics
    t = SampleTable()
    # two piles of two photos each (a photo pair shares one weight): ratios a: 0.04, 0.05; b: 0.045, 0.06
    for i, (pile, n_box, w) in enumerate([("a", 10, 0.40), ("a", 10, 0.50), ("b", 20, 0.90), ("b", 5, 0.30)]):
        t.add(f"w{i}.jpg", [box(j * 30, 0) for j in range(n_box)],
              {"crop": "wheat", "total_weight_g": w, "pile_id": pile})
    t.add("s0.jpg", [box(0, 0)], {"crop": "sunflower", "total_weight_g": 0.05})           # one pile: no spread
    m = CountTimesConstant().fit(t)
    pile_means = [(0.04 + 0.05) / 2, (0.045 + 0.06) / 2]
    assert m.spreads["wheat"] == {"sd": pytest.approx(statistics.stdev(pile_means)), "n": 2, "photos": 4}
    assert "sunflower" not in m.spreads                                                  # n = 1 pile -> nothing to estimate
    assert m.spreads["default"]["n"] == 3 and m.spreads["default"]["photos"] == 5 and m.spreads["default"]["sd"] > 0
    m.save(tmp_path / "m.json")
    m2 = WeightModel.load(tmp_path / "m.json")
    assert m2.spreads == m.spreads and m2.constants == m.constants
    old = CountTimesConstant()                                                           # saved before task 3
    old.set_params({"constants": {"default": 0.04}})
    assert old.spreads == {}
    solo = SampleTable()                                                                 # no pile_id: each photo is a pile
    for i, w in enumerate([0.40, 0.50]):
        solo.add(f"p{i}.jpg", [box(j * 30, 0) for j in range(10)], {"crop": "wheat", "total_weight_g": w})
    assert CountTimesConstant().fit(solo).spreads["wheat"]["n"] == 2


def test_predict_gives_95_percent_prediction_interval():
    import statistics
    from detkit.weight.models import t_factor
    t = SampleTable()
    ratios = [0.040, 0.044, 0.048, 0.052, 0.056, 0.060]                    # one pile each, 10 kernels
    for i, r in enumerate(ratios):
        t.add(f"w{i}.jpg", [box(j * 30, 0) for j in range(10)], {"crop": "wheat", "total_weight_g": 10 * r,
                                                                    "pile_id": f"p{i}"})
    m = CountTimesConstant().fit(t)
    n, sd = 6, statistics.stdev(ratios)
    e = m.predict({"count": 20}, "wheat")
    u = t_factor(n - 1) * math.sqrt(1 + 1 / n) * sd * 20
    assert t_factor(5) == pytest.approx(2.571)
    assert e.weight_g == pytest.approx(20 * statistics.fmean(ratios))
    assert e.low_g == pytest.approx(e.weight_g - u) and e.high_g == pytest.approx(e.weight_g + u)
    assert e.low_g < e.weight_g < e.high_g
    wide = m.predict({"count": 40}, "wheat")                                # bounds scale with the feature
    assert wide.high_g - wide.low_g == pytest.approx(2 * (e.high_g - e.low_g))
    other = m.predict({"count": 20}, "corn")                                # unknown crop: default constant + spread
    assert other.low_g is not None and "default" in other.note
    assert m.predict({"count": 0}, "wheat").low_g == 0.0
    one = CountTimesConstant().fit(SampleTable())                           # nothing fitted: constants only
    e1 = one.predict({"count": 10}, "wheat")
    assert e1.low_g is None and e1.high_g is None and "no bounds" in e1.note


def test_count_times_constant(tmp_path):
    m = CountTimesConstant({"default": 0.04, "wheat": 0.05})
    assert m.predict({"count": 10}, "wheat").weight_g == pytest.approx(0.5)
    e = m.predict({"count": 10}, "corn")
    assert e.weight_g == pytest.approx(0.4) and "default" in e.note
    t = SampleTable.from_project(make_project(tmp_path)).with_weight()
    m.fit(t)
    assert 0.04 < m.constants["wheat"] < 0.07
    ev = evaluate_model(m, t)
    assert ev["n"] == 4 and ev["per_crop"]["wheat"]["n"] == 4 and len(ev["rows"]) == 4
    assert abs(ev["bias_g"]) < 0.02
    m.save(tmp_path / "m.json")
    m2 = WeightModel.load(tmp_path / "m.json")
    assert m2.constants == m.constants


def test_area_times_constant(tmp_path):
    assert set(MODELS) == {"count_x_constant", "area_x_constant", "kernel_area_x_constant"}
    m = AreaTimesConstant()
    with pytest.raises(ValueError, match="calibrate"):                  # no default: camera-dependent
        m.predict({"total_area_px": 100}, "wheat")
    t = SampleTable.from_project(make_project(tmp_path)).with_weight()
    m.fit(t)
    f = t.rows[0].features
    assert m.predict(f, "wheat").weight_g == pytest.approx(f["total_area_px"] * m.constants["wheat"])
    assert 0.04 / 200 < m.constants["wheat"] < 0.07 / 200                # boxes are 10x20 px
    assert WeightModel.from_dict(m.to_dict()).constants == m.constants
    with pytest.raises(KeyError):
        get_model("nope")


def test_config(tmp_path):
    cfg = load_config(tmp_path)
    assert cfg["model"] == "count_x_constant" and cfg["constants"]["wheat"] > 0
    cfg["constants"]["wheat"] = 0.06
    save_config(tmp_path, cfg)
    assert load_config(tmp_path)["constants"]["wheat"] == 0.06
    assert "sunflower" in load_config(tmp_path)["constants"]


def test_calibrate_saves_spread_and_model_gives_bounds(tmp_path):
    from detkit.cmd_weight import configured_model
    make_project(tmp_path)
    w = str(tmp_path)
    assert load_config(w)["spreads"] == {}                                   # default: no spread, no bounds
    assert main(["weight", "calibrate", "--workdir", w]) == 0
    cfg = load_config(w)
    sp = cfg["spreads"]["constants"]
    assert sp["wheat"]["n"] == 4 and sp["wheat"]["sd"] > 0 and "sunflower" not in sp   # sunflower: no weighed photo
    m = configured_model(cfg, "count_x_constant")                            # rebuilt from the config, as the app does
    e = m.predict({"count": 10}, "wheat")
    assert m.spreads == sp and e.low_g is not None and e.low_g < e.weight_g < e.high_g
    assert configured_model(cfg, "area_x_constant").spreads == {}            # other models keep their own spreads


def test_calibrate_warns_about_piles_with_different_weights(tmp_path, capsys):
    p = make_project(tmp_path)
    st = p.store()
    for f, w in (("w0.jpg", 0.10), ("w1.jpg", 0.70)):                         # same pile, one weight is a typo
        boxes, meta = st.get_image(f)
        st.upsert_image(f, 1000, 1000, boxes, {**meta, "pile_id": "A", "total_weight_g": w}, reviewed=True)
    t = SampleTable.from_project(p)
    bad = t.inconsistent_piles()
    assert list(bad) == ["A"] and sorted(s.file for s in bad["A"]) == ["w0.jpg", "w1.jpg"]
    assert main(["weight", "calibrate", "--workdir", str(tmp_path)]) == 0
    err = capsys.readouterr().err
    assert "WARNING: pile A" in err and "w1.jpg 0.7 g" in err
    for f in ("w0.jpg", "w1.jpg"):                                            # fixed: no warning
        boxes, meta = st.get_image(f)
        st.upsert_image(f, 1000, 1000, boxes, {**meta, "total_weight_g": 0.10}, reviewed=True)
    assert SampleTable.from_project(p).inconsistent_piles() == {}


def test_cli(tmp_path, capsys):
    make_project(tmp_path)
    w = str(tmp_path)
    assert main(["weight", "predict", "--workdir", w, "--out", str(tmp_path / "p.csv")]) == 0
    out = capsys.readouterr().out
    assert "w1.jpg" in out and (tmp_path / "p.csv").exists()
    assert main(["weight", "predict", "--workdir", w, "--model", "area_x_constant"]) == 1   # not calibrated
    assert main(["weight", "calibrate", "--workdir", w]) == 0
    assert (tmp_path / "weight_config.json").exists()
    assert main(["weight", "calibrate", "--workdir", w, "--model", "area_x_constant"]) == 0
    cfg = load_config(w)
    assert cfg["model"] == "area_x_constant" and cfg["area_constants"]["wheat"] > 0 and cfg["constants"]["wheat"] > 0
    assert main(["weight", "predict", "--workdir", w]) == 0
