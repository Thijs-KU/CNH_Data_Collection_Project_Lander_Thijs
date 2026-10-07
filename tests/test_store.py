from detkit import coco as C
from detkit.project import FLOWS, Project
from detkit.store import Store


def test_store_upsert_roundtrip_and_replace(tmp_path):
    st = Store(tmp_path, ["wheat", "sunflower"])
    st.upsert_image("a.jpg", 100, 80, [
        {"bbox": [1, 2, 11, 12], "class": "wheat", "origin": "rfdetr", "score": 0.9},
        {"bbox": [20, 20, 40, 50], "class": "sunflower", "origin": "manual", "weight_g": 0.04}],
        {"crop": "wheat", "total_weight_g": 1.2, "bogus": 1}, proposer="rfdetr")
    boxes, meta = st.get_image("a.jpg")
    assert [b["class"] for b in boxes] == ["wheat", "sunflower"] and boxes[1]["weight_g"] == 0.04
    assert meta["crop"] == "wheat" and "bogus" not in meta and meta["reviewed"] and meta["proposer"] == "rfdetr"
    st.upsert_image("a.jpg", 100, 80, [{"bbox": [5, 5, 9, 9], "class": "wheat"}], {"pile_id": "x"})
    boxes, meta = st.get_image("a.jpg")
    assert len(boxes) == 1 and meta["total_weight_g"] == 1.2 and meta["pile_id"] == "x"   # meta merged
    assert st.reviewed_files() == ["a.jpg"]


def test_store_new_class_appended_not_reordered(tmp_path):
    Store(tmp_path, ["wheat"]).upsert_image("a.jpg", 10, 10, [{"bbox": [0, 0, 5, 5], "class": "wheat"}])
    st = Store(tmp_path, ["wheat", "corn"])
    st.upsert_image("b.jpg", 10, 10, [{"bbox": [0, 0, 5, 5], "class": "corn"}])
    coco = st.load_coco()
    assert C.class_names(coco) == ["wheat", "corn"] and C.count_per_class(coco) == {"wheat": 1, "corn": 1}


def test_project_flows_and_old_json(tmp_path):
    assert set(FLOWS) == {"hf", "cpu", "gpu"} and FLOWS["cpu"]["train_workers"] == 0
    p = Project(workdir=tmp_path / "w", **FLOWS["gpu"], flow="gpu")
    p.save()
    (tmp_path / "w" / "project.json").write_text('{"classes": ["seed"], "removed_field": 1}')
    q = Project.load(tmp_path / "w")
    assert q.classes == ["seed"] and q.flow == "hf" and q.store().classes == ["seed"]
