from detkit import coco as C
from detkit import merge


def test_iou_ioma():
    assert merge.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert merge.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0
    assert merge.ioma([0, 0, 10, 10], [2, 2, 4, 4]) == 1  # nested


def test_nms_prefers_score_then_area():
    boxes = [[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]]
    assert merge.nms(boxes, [0.5, 0.9, 0.4]) == [1, 2]
    assert merge.nms(boxes)[0] == 0  # largest area first


def test_nms_per_class_does_not_cross_classes():
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
    assert merge.nms_per_class(boxes, [0, 1]) == [0, 1]
    assert merge.nms_per_class(boxes, [0, 0]) == [0]


def test_tiles_to_source_offsets_and_dedup():
    man = {"sources": {"p.jpg": {"width": 2000, "height": 1000, "crop": None, "tile": 800}},
           "tiles": [{"file": "p_x0_y0.jpg", "source": "p.jpg", "x": 0, "y": 0, "w": 800, "h": 800},
                     {"file": "p_x600_y0.jpg", "source": "p.jpg", "x": 600, "y": 0, "w": 800, "h": 800}]}
    t = C.new_coco(["a", "b"])
    i0 = C.add_image(t, "p_x0_y0.jpg", 800, 800)
    i1 = C.add_image(t, "p_x600_y0.jpg", 800, 800)
    C.add_box(t, i0, 0, [650, 100, 750, 200])   # same object seen in both tiles
    C.add_box(t, i1, 0, [50, 100, 150, 200])    # -> x+600 = [650,100,750,200]
    C.add_box(t, i1, 1, [50, 100, 150, 200])    # other class, same place: kept
    out = merge.tiles_to_source(t, man)
    assert out["images"][0]["file_name"] == "p.jpg" and out["images"][0]["width"] == 2000
    assert C.count_per_class(out) == {"a": 1, "b": 1}
    assert out["annotations"][0]["bbox"] == [650.0, 100.0, 100.0, 100.0]
