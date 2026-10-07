import json
import math

import pytest

from detkit.cli import main
from detkit.project import Project
from detkit.weight import (MODELS, AreaTimesConstant, CountTimesConstant, SampleTable, WeightModel,
                           evaluate_model, get_model, image_features, load_config, save_config)


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
    assert set(MODELS) == {"count_x_constant", "area_x_constant"}
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
