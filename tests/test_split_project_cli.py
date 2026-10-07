import json

import pytest
from detkit import cli, coco as C
from detkit.project import Project, load_env, parse_classes
from detkit.split import build_dataset


def _tiles_coco(tmp_path, sources=("a.jpg", "b.jpg", "c.jpg", "d.jpg")):
    from PIL import Image
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    man = {"sources": {}, "tiles": []}
    coco = C.new_coco(["x", "y"])
    for s in sources:
        man["sources"][s] = {"width": 100, "height": 100, "crop": None, "tile": 50}
        for k in range(2):
            f = f"{s[:-4]}_{k}.jpg"
            Image.new("RGB", (50, 50)).save(tiles / f)
            man["tiles"].append({"file": f, "source": s, "x": 0, "y": 0, "w": 50, "h": 50})
            i = C.add_image(coco, f, 50, 50)
            C.add_box(coco, i, k % 2, [5, 5, 30, 30])
    return coco, man, tiles


def test_split_by_source_no_leak(tmp_path):
    coco, man, tiles = _tiles_coco(tmp_path)
    rep = build_dataset(coco, man, tiles, tmp_path / "out", val_sources=["b.jpg"])
    tr = {i["file_name"] for i in C.load(tmp_path / "out/train/labels.json")["images"]}
    va = {i["file_name"] for i in C.load(tmp_path / "out/val/labels.json")["images"]}
    assert va == {"b_0.jpg", "b_1.jpg"} and not (tr & va) and len(tr) == 6
    assert (tmp_path / "out/val/images/b_0.jpg").exists() and rep["warnings"] == []


def test_split_frac_is_seeded_and_warns_on_missing_class(tmp_path):
    coco, man, tiles = _tiles_coco(tmp_path)
    r1 = build_dataset(coco, man, tiles, tmp_path / "o1", val_frac=0.5, seed=1)
    r2 = build_dataset(coco, man, tiles, tmp_path / "o2", val_frac=0.5, seed=1)
    assert r1["val_sources"] == r2["val_sources"] and len(r1["val_sources"]) == 2
    with pytest.raises(ValueError):
        build_dataset(coco, man, tiles, tmp_path / "o3", val_sources=["nope.jpg"])
    with pytest.raises(ValueError):
        build_dataset(coco, man, tiles, tmp_path / "o4", val_sources=["a.jpg", "b.jpg", "c.jpg", "d.jpg"])


def test_project_roundtrip_and_classes(tmp_path):
    p = Project(workdir=tmp_path / "w", classes=parse_classes("a, b"), tile=640)
    p.save()
    q = Project.load(tmp_path / "w")
    assert q.classes == ["a", "b"] and q.tile == 640 and q.tiles_coco.name == "tiles_coco.json"
    with pytest.raises(ValueError):
        parse_classes("a,A")
    with pytest.raises(FileNotFoundError):
        Project.load(tmp_path / "missing")


def test_load_env_does_not_override(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("# c\nHF_TOKEN=abc\nOTHER='x y'\n")
    monkeypatch.setenv("HF_TOKEN", "keep")
    monkeypatch.delenv("OTHER", raising=False)
    assert load_env(tmp_path) == tmp_path / ".env"
    import os
    assert os.environ["HF_TOKEN"] == "keep" and os.environ["OTHER"] == "x y"



def test_cli_student_flow_import_review_split(tmp_path, photos):
    """init -> import-photos -> (review in the app = Store.upsert_image) -> split --from-reviewed."""
    w = str(tmp_path / "run")
    assert cli.main(["init", "--workdir", w, "--classes", "wheat,sunflower", "--tile", "800"]) == 0
    assert cli.main(["import-photos", "--workdir", w, "--images", str(photos), "--crop", "wheat"]) == 0
    p = Project.load(w)
    st = p.store()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        from PIL import Image
        with Image.open(p.photos_dir / name) as im:
            W, H = im.size
        st.upsert_image(name, W, H, [{"bbox": [100, 100, 160, 160], "class": "wheat"}], reviewed=True)
    assert cli.main(["split", "--workdir", w, "--from-reviewed", "--val-sources", "b.jpg"]) == 0
    wd = tmp_path / "run"
    assert (wd / "coco/val/labels.json").exists() and (wd / "coco/train/labels.json").exists()
    cats = C.load(wd / "coco/train/labels.json")["categories"]
    assert [c["name"] for c in cats] == ["wheat", "sunflower"]        # multi-class order preserved
