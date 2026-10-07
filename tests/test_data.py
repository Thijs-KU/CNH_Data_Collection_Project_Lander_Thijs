import json

import pytest
from detkit import cli, coco as C
from detkit.cmd_data import import_photos, read_meta_file, retile
from detkit.project import Project


def _proj(tmp_path, classes="wheat,sunflower"):
    w = str(tmp_path / "run")
    assert cli.main(["init", "--workdir", w, "--classes", classes]) == 0
    return w




def test_import_photos(tmp_path, photos, capsys):
    w = _proj(tmp_path)
    csvf = tmp_path / "m.csv"
    csvf.write_text("file,crop,total_weight_g,manual_count\na.jpg,sunflower,12.5,40\n")
    assert cli.main(["import-photos", "--workdir", w, "--images", str(photos), "--crop", "wheat",
                     "--meta", str(csvf)]) == 0
    p = Project.load(w)
    meta = p.store().load_meta()
    assert set(meta) == {"a.jpg", "b.jpg", "c.jpg"} and (p.photos_dir / "a.jpg").exists()
    assert meta["a.jpg"]["crop"] == "sunflower" and meta["a.jpg"]["total_weight_g"] == 12.5
    assert meta["a.jpg"]["manual_count"] == 40 and meta["b.jpg"]["crop"] == "wheat"
    assert meta["b.jpg"]["reviewed"] is False
    assert len(p.store().load_coco()["images"]) == 3 and p.store().reviewed_files() == []
    # re-import: identical skipped, nothing changes
    r = import_photos(p, photos)
    assert r["copied"] == [] and len(r["identical_skipped"]) == 3 and r["registered"] == []
    # differing content refuses, then --force overwrites
    from PIL import Image
    d2 = tmp_path / "p2"
    d2.mkdir()
    Image.new("RGB", (10, 10)).save(d2 / "a.jpg")
    before = (p.photos_dir / "a.jpg").read_bytes()
    assert cli.main(["import-photos", "--workdir", w, "--images", str(d2)]) == 1
    assert (p.photos_dir / "a.jpg").read_bytes() == before
    assert cli.main(["import-photos", "--workdir", w, "--images", str(d2), "--force"]) == 0
    assert (p.photos_dir / "a.jpg").read_bytes() != before


def test_import_photos_crop_validation_and_add_class(tmp_path, photos):
    w = _proj(tmp_path)
    assert cli.main(["import-photos", "--workdir", w, "--images", str(photos), "--crop", "corn"]) == 1
    assert not (tmp_path / "run/photos").exists()              # nothing copied on failure
    assert cli.main(["import-photos", "--workdir", w, "--images", str(photos), "--crop", "corn",
                     "--add-class", "corn"]) == 0
    p = Project.load(w)
    assert p.classes == ["wheat", "sunflower", "corn"]
    assert p.store().load_meta()["a.jpg"]["crop"] == "corn"


def test_read_meta_json(tmp_path):
    f = tmp_path / "m.json"
    f.write_text(json.dumps([{"file": "x/a.jpg", "crop": "wheat", "manual_count": "11", "bogus": 1}]))
    assert read_meta_file(f) == {"a.jpg": {"crop": "wheat", "manual_count": 11}}


def _setup_reviewed(tmp_path, photos):
    w = _proj(tmp_path)
    cli.main(["import-photos", "--workdir", w, "--images", str(photos), "--crop", "wheat"])
    cli.main(["tile", "--workdir", w])
    st = Project.load(w).store()
    a_boxes = [
        {"bbox": [780, 100, 880, 200], "class": "wheat", "origin": "manual"},     # straddles x=800
        {"bbox": [450, 100, 750, 200], "class": "wheat", "origin": "vlm", "score": 0.5},  # in 2 tiles
        {"bbox": [0, 0, 5, 50], "class": "wheat"},                                # too thin
        {"bbox": [700, 100, 900, 200], "class": "sunflower"},                     # < 60% anywhere except
    ]
    st.upsert_image("a.jpg", 1600, 1200, a_boxes, reviewed=True)
    st.upsert_image("c.jpg", 1000, 1400, [{"bbox": [100, 100, 200, 200], "class": "sunflower"}],
                    reviewed=True)
    st.upsert_image("b.jpg", 1600, 1200, [{"bbox": [100, 100, 200, 200], "class": "wheat"}],
                    reviewed=False)
    return w, Project.load(w)


def test_retile_clipping_math_and_straddling_box(tmp_path, photos):
    w, p = _setup_reviewed(tmp_path, photos)
    st = retile(p)
    tc = C.load(p.tiles_coco)
    assert st["photos"] == 2 and "b_x0_y0.jpg" not in {i["file_name"] for i in tc["images"]}
    assert C.class_names(tc) == ["wheat", "sunflower"]
    ids = {i["file_name"]: i["id"] for i in tc["images"]}

    def boxes(tile):
        return [(a["category_id"], a["bbox"], a.get("origin")) for a in tc["annotations"]
                if a["image_id"] == ids[tile]]

    # straddler [780..880]: 20% in tile x0 (dropped), 100% in x400, 80% in x800 (clipped to 0..80)
    assert not any(b[1][0] == 780 for b in boxes("a_x0_y0.jpg"))
    assert (0, [380.0, 100.0, 100.0, 100.0], "manual") in boxes("a_x400_y0.jpg")
    assert (0, [0.0, 100.0, 80.0, 100.0], "manual") in boxes("a_x800_y0.jpg")
    # box wholly inside two overlapping tiles is kept in both, shifted by each offset
    assert (0, [450.0, 100.0, 300.0, 100.0], "vlm") in boxes("a_x0_y0.jpg")
    assert (0, [50.0, 100.0, 300.0, 100.0], "vlm") in boxes("a_x400_y0.jpg")
    # sliver below min_px never survives; sunflower box [700..900] is 50% in x0/x800 -> only x400
    allb = [b for t in ids for b in boxes(t)]
    assert not any(b[1][2] == 5 for b in allb)
    assert [b for b in boxes("a_x400_y0.jpg") if b[0] == 1] and not [b for b in boxes("a_x0_y0.jpg") if b[0] == 1]
    assert st["boxes_never_kept"] == 1 and st["per_class"]["sunflower"] == 2  # a's + c's
    # lower thresholds keep more
    retile(p, min_visible=0.4, min_px=4)
    tc2 = C.load(p.tiles_coco)
    assert len(tc2["annotations"]) > len(tc["annotations"])


def test_retile_include_unreviewed(tmp_path, photos):
    w, p = _setup_reviewed(tmp_path, photos)
    retile(p, include_unreviewed=True)
    assert any(i["file_name"].startswith("b_") for i in C.load(p.tiles_coco)["images"])


def test_retile_requires_manifest(tmp_path):
    w = _proj(tmp_path)
    with pytest.raises(FileNotFoundError):
        retile(Project.load(w))
    assert cli.main(["retile", "--workdir", w]) == 1


def test_end_to_end_reviewed_retile_split(tmp_path, photos, capsys):
    w, p = _setup_reviewed(tmp_path, photos)
    assert cli.main(["split", "--workdir", w, "--from-reviewed", "--val-sources", "c.jpg"]) == 0
    tr = C.load(p.coco_dir / "train/labels.json")
    va = C.load(p.coco_dir / "val/labels.json")
    assert {i["file_name"][0] for i in tr["images"]} == {"a"} and {i["file_name"][0] for i in va["images"]} == {"c"}
    assert C.class_names(tr) == ["wheat", "sunflower"] and len(va["annotations"]) == 1
    err = capsys.readouterr().err
    assert "class 'wheat' has 0 boxes in val" in err            # warning surfaced
    assert cli.main(["split", "--workdir", w, "--from-reviewed", "--include-unreviewed",
                     "--val-sources", "c.jpg"]) == 0
    assert any(i["file_name"].startswith("b_") for i in C.load(p.coco_dir / "train/labels.json")["images"])


def test_retile_skips_unreviewed_empty_photos_but_keeps_reviewed_negatives(tmp_path, photos):
    w, p = _setup_reviewed(tmp_path, photos)
    st = p.store()
    st.upsert_image("b.jpg", 1600, 1200, [], reviewed=False)                  # unknown content
    st.upsert_image("c.jpg", 1000, 1400, [], reviewed=True)                   # verified empty = negative
    stats = retile(p, include_unreviewed=True)
    files = {i["file_name"].split("_x")[0] for i in C.load(p.tiles_coco)["images"]}
    assert "b" not in files and "c" in files and "a" in files
    assert stats["photos_skipped_unlabelled"] == ["b.jpg"]