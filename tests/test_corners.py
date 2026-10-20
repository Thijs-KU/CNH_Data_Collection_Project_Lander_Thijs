"""Manual A4 warp: corner ordering, the warp itself, and the Corners tab logic."""

import json

import cv2
import numpy as np
import pytest
from PIL import Image

from detkit.app import corners as K
from detkit.preprocess import CANVAS_SIZE, SCALE_MM_PER_PX, brightness, normalize_brightness, order_corners, warp_to_canvas
from detkit.project import Project

W, H = CANVAS_SIZE


def pattern() -> np.ndarray:
    """Smooth, asymmetric BGR test image (so a wrong orientation or flip is visible)."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.dstack([127 + 100 * np.sin(x / 150), 127 + 100 * np.cos(y / 200), 255 * x / W]).astype(np.uint8)


def photo_of(pat: np.ndarray, quad: list[list[float]], size=(3000, 2250)):
    """A fake photo in which the pattern occupies ``quad`` (TL, TR, BR, BL of the pattern)."""
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), np.float32(quad))
    bgr = cv2.warpPerspective(pat, M, size, borderValue=(60, 60, 60))
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


QUAD = [[900, 300], [2300, 400], [2200, 2000], [700, 1900]]           # perspective-ish, upright


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (2, 0, 3, 1), (3, 2, 1, 0), (1, 3, 0, 2)])
def test_order_corners_any_click_order(order):
    got = order_corners([QUAD[i] for i in order])
    assert np.allclose(got, QUAD)


def test_order_corners_short_edge_on_top_when_sheet_lies_landscape():
    land = [[500, 600], [2500, 600], [2500, 1700], [500, 1700]]       # 2000 wide, 1100 tall
    got = order_corners(land)
    assert np.linalg.norm(got[1] - got[0]) <= np.linalg.norm(got[3] - got[0])
    assert tuple(got[0]) == (500, 1700)                                # nearer the top-left of the two candidates


def test_order_corners_rejects_bad_input():
    with pytest.raises(ValueError, match="exactly 4"):
        order_corners(QUAD[:3])
    with pytest.raises(ValueError, match="convex"):
        order_corners([[0, 0], [100, 0], [50, 10], [50, 100]])


def test_warp_recovers_the_flat_sheet():
    pat = pattern()
    img = photo_of(pat, QUAD)
    out, info = warp_to_canvas(img, [QUAD[2], QUAD[0], QUAD[3], QUAD[1]])      # clicked in a shuffled order
    assert out.size == CANVAS_SIZE
    got = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR).astype(np.float32)
    inner = (slice(10, H - 10), slice(10, W - 10))
    assert np.abs(got[inner] - pat.astype(np.float32)[inner]).mean() < 4
    assert info["canvas"] == [W, H] and np.allclose(info["corners"], QUAD)
    assert info["mm_per_px"] == pytest.approx([210 / W, 297 / H])
    assert SCALE_MM_PER_PX == pytest.approx((210 / W * 297 / H) ** 0.5)


# ---- tab logic ---------------------------------------------------------------------------------
@pytest.fixture
def project(tmp_path):
    p = Project(workdir=tmp_path / "w", classes=["wheat"])
    p.save()
    p.photos_dir.mkdir()
    photo_of(pattern(), QUAD).save(p.photos_dir / "IMG_1.jpg", quality=95)        # a RAW project photo
    photo_of(pattern(), QUAD).save(p.photos_dir / "IMG_2.png")
    p.store().upsert_image("IMG_1.jpg", 3000, 2250, [{"bbox": [10, 10, 50, 50], "class": "wheat"}],
                           {"crop": "wheat", "pile_id": "P1", "total_weight_g": 7.5}, reviewed=True)
    p.store().upsert_image("IMG_2.png", 3000, 2250, [], {}, reviewed=False)
    return p


def test_clicks_map_back_to_original_pixels(project):
    raw = project.photos_dir / "IMG_1.jpg"
    disp, scale = K.load_display(raw, maxside=1500)
    assert scale == pytest.approx(2.0) and disp.size == (1500, 1125)
    s = {**K.new_state(), "name": "IMG_1.jpg", "path": str(raw), "scale": scale}
    for x, y in [(10, 20), (30, 40), (50, 60), (70, 80), (90, 100)]:
        s = K.add_point(s, x, y)
    assert s["pts"] == [[20, 40], [60, 80], [100, 120], [140, 160]]       # 5th click ignored
    assert K.shown_points(s)[0] == [10, 20]
    assert len(K.undo(s)["pts"]) == 3 and K.clear(s)["pts"] == []
    assert K.draw_points(disp, K.shown_points(s)).size == disp.size
    assert K.preview(K.undo(s)) is None and K.preview({**s, "pts": QUAD}).size == CANVAS_SIZE


def test_list_matches_the_review_list_and_marks_warped(project):
    from detkit.app import logic as L
    r = K.rows(project)
    assert [x[0] for x in r] == L.list_photos(project.photos_dir) == ["IMG_1.jpg", "IMG_2.png"]
    assert r[0][:6] == ["IMG_1.jpg", "raw", "reviewed", 1, "3000x2250", "-"] and isinstance(r[0][6], int)
    assert r[1][:4] == ["IMG_2.png", "raw", "todo", 0]
    K.warp_photo(project, {**K.new_state(), "name": "IMG_2.png", "pts": QUAD})
    assert K.rows(project)[1][:6] == ["IMG_2.png", "warped", "todo", 0, "900x1200", "-"]


def test_warp_photo_in_place(project):
    raw_bytes = (project.photos_dir / "IMG_1.jpg").read_bytes()
    msg = K.warp_photo(project, {**K.new_state(), "name": "IMG_1.jpg", "pts": QUAD})
    assert "1 saved boxes" in msg
    assert Image.open(project.photos_dir / "IMG_1.jpg").size == CANVAS_SIZE            # same name, new frame
    assert (project.workdir / "photos_original" / "IMG_1.jpg").read_bytes() == raw_bytes
    store = project.store()
    boxes, meta = store.get_image("IMG_1.jpg")
    assert boxes == [] and meta["reviewed"] is False
    assert meta["crop"] == "wheat" and meta["pile_id"] == "P1" and meta["total_weight_g"] == 7.5   # metadata kept
    assert [(i["width"], i["height"]) for i in store.load_coco()["images"] if i["file_name"] == "IMG_1.jpg"] == [CANVAS_SIZE]
    rec = json.loads((project.workdir / "meta" / K.WARPS_FILE).read_text())["IMG_1.jpg"]
    assert rec["source"] == "IMG_1.jpg" and np.allclose(rec["corners"], QUAD)
    assert rec["discarded"] == [{"reviewed": True, "boxes": [{"bbox": [10.0, 10.0, 50.0, 50.0], "class": "wheat",
                                                               "origin": None, "score": None, "weight_g": None,
                                                               "box_notes": None}]}]


def test_redo_starts_from_the_original_and_needs_four_corners(project):
    K.warp_photo(project, {**K.new_state(), "name": "IMG_1.jpg", "pts": QUAD})
    assert K.source_of(project, "IMG_1.jpg") == project.workdir / "photos_original" / "IMG_1.jpg"
    again = [[x + 5, y] for x, y in QUAD]
    K.warp_photo(project, {**K.new_state(), "name": "IMG_1.jpg", "pts": again})            # redo
    assert Image.open(project.photos_dir / "IMG_1.jpg").size == CANVAS_SIZE
    assert len(list((project.workdir / "photos_original").iterdir())) == 1                 # no second copy
    rec = json.loads((project.workdir / "meta" / K.WARPS_FILE).read_text())["IMG_1.jpg"]
    assert np.allclose(rec["corners"], again) and len(rec["discarded"]) == 1
    before = (project.photos_dir / "IMG_2.png").read_bytes()
    for bad in ({"pts": QUAD[:2]}, {"pts": [[0, 0], [100, 0], [50, 10], [50, 100]]}):
        with pytest.raises(ValueError):
            K.warp_photo(project, {**K.new_state(), "name": "IMG_2.png", **bad})
    assert (project.photos_dir / "IMG_2.png").read_bytes() == before and K.load_warps(project).get("IMG_2.png") is None


def test_scale_recorded_when_the_property_exists(project):
    project.extra_image_fields.append({"key": "scale_mm_per_px", "kind": "float", "unit": "mm"})
    K.warp_photo(project, {**K.new_state(), "name": "IMG_2.png", "pts": QUAD})
    assert project.store().load_meta()["IMG_2.png"]["scale_mm_per_px"] == pytest.approx(SCALE_MM_PER_PX, abs=1e-6)


def test_import_photos_recognises_a_photo_warped_in_place(project, tmp_path):
    from detkit.cmd_data import import_photos
    src = tmp_path / "raw"
    src.mkdir()
    shutil_copy = (project.photos_dir / "IMG_2.png").read_bytes()
    (src / "IMG_2.png").write_bytes(shutil_copy)
    K.warp_photo(project, {**K.new_state(), "name": "IMG_2.png", "pts": QUAD})
    r = import_photos(project, src)
    assert r["identical_skipped"] == ["IMG_2.png"] and r["overwritten"] == []             # same source: left alone


def test_tab_builds(project):
    gr = pytest.importorskip("gradio")
    from detkit.app.corners_ui import build_corners_tab
    with gr.Blocks():
        with gr.Tab("Preprocessing") as t:
            build_corners_tab(project, project.workdir, t)


# ---- brightness / contrast ----------------------------------------------------------------------------------
def test_normalize_brightness_hits_the_target_and_keeps_the_size():
    rng = np.random.default_rng(1)
    for base in (80, 140, 230):
        a = np.clip(rng.normal(base, 12, (240, 320, 3)), 0, 255).astype(np.uint8)
        a[:40, :60] = 30                                                  # a dark "kernel" patch
        im = Image.fromarray(a)
        for con in (0.0, 0.7):
            out, info = normalize_brightness(im, 200, con)
            assert out.size == im.size and abs(info["after"] - 200) <= 1 and abs(brightness(out) - 200) <= 2
    with pytest.raises(ValueError):
        normalize_brightness(im, 300)
    with pytest.raises(ValueError):
        normalize_brightness(im, 200, 2)


def test_contrast_enhancement_widens_the_spread():
    y, x = np.mgrid[0:200, 0:300]
    flat = np.dstack([120 + x // 30 + 0 * y] * 3).astype(np.uint8)         # nearly flat grey ramp
    lo = np.asarray(normalize_brightness(Image.fromarray(flat), 200, 0)[0].convert("L")).std()
    hi = np.asarray(normalize_brightness(Image.fromarray(flat), 200, 1)[0].convert("L")).std()
    assert hi > lo


def test_normalize_photo_keeps_boxes_and_is_redone_from_the_original(project):
    raw = (project.photos_dir / "IMG_1.jpg").read_bytes()
    before = project.store().get_image("IMG_1.jpg")
    msg = K.normalize_photo(project, "IMG_1.jpg", 180, 0.0)
    assert "IMG_1.jpg" in msg
    assert project.store().get_image("IMG_1.jpg") == before                  # boxes + metadata untouched
    assert Image.open(project.photos_dir / "IMG_1.jpg").size == (3000, 2250)
    assert abs(brightness(Image.open(project.photos_dir / "IMG_1.jpg")) - 180) <= 2
    assert (project.workdir / "photos_original" / "IMG_1.jpg").read_bytes() == raw
    K.normalize_photo(project, "IMG_1.jpg", 150, 0.0)                         # other settings: no stacking
    assert abs(brightness(Image.open(project.photos_dir / "IMG_1.jpg")) - 150) <= 2
    assert K.rows(project)[0][1] == "raw" and K.rows(project)[0][5] == "yes"
    assert "removed" in K.reset_photo_brightness(project, "IMG_1.jpg").lower()
    assert "normalize" not in K.load_warps(project)["IMG_1.jpg"]
    assert K.rows(project)[0][5] == "-"


def test_warp_then_normalize_then_rewarp_keeps_the_brightness(project):
    K.warp_photo(project, {**K.new_state(), "name": "IMG_2.png", "pts": QUAD})
    K.normalize_photo(project, "IMG_2.png", 190, 0.0)
    assert K.rows(project)[1][1:2] == ["warped"] and abs(K.rows(project)[1][6] - 190) <= 3
    again = [[x + 5, y] for x, y in QUAD]
    K.warp_photo(project, {**K.new_state(), "name": "IMG_2.png", "pts": again})
    rec = K.load_warps(project)["IMG_2.png"]
    assert np.allclose(rec["corners"], again) and rec["normalize"]["target"] == 190
    assert abs(brightness(Image.open(project.photos_dir / "IMG_2.png")) - 190) <= 3
    img, msg = K.normalized_preview(project, "IMG_2.png", 150, 0.3)
    assert img.size == CANVAS_SIZE and "->" in msg
    assert abs(brightness(Image.open(project.photos_dir / "IMG_2.png")) - 190) <= 3   # a preview stores nothing


def test_normalize_all(project):
    assert K.normalize_all(project, 200, 0.0).startswith("Normalized 2 of 2")
    assert [abs(r[6] - 200) <= 2 for r in K.rows(project)] == [True, True]


def test_reimport_of_a_normalized_photo_is_recognised(project, tmp_path):
    from detkit.cmd_data import import_photos
    src = tmp_path / "raw"
    src.mkdir()
    (src / "IMG_2.png").write_bytes((project.photos_dir / "IMG_2.png").read_bytes())
    K.normalize_photo(project, "IMG_2.png", 150, 0.0)
    r = import_photos(project, src)
    assert r["identical_skipped"] == ["IMG_2.png"] and r["overwritten"] == []
