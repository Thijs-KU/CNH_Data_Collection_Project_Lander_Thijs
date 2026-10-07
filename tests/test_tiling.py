import json

from detkit import tiling


def test_axis_starts_cover_and_overlap():
    for lo, hi, t in [(0, 2000, 800), (100, 1900, 900), (0, 800, 800), (0, 500, 800)]:
        s = tiling.axis_starts(lo, hi, t, 0.2)
        assert s[0] == lo
        assert min(hi, s[-1] + t) >= hi if hi - lo > t else len(s) == 1
        for a, b in zip(s, s[1:]):
            assert b - a <= t * 0.8 + 1  # overlap at least 20% (allow rounding)


def test_plan_tiles_counts_and_clamp():
    assert len(tiling.plan_tiles((2000, 2000), (0, 0, 2000, 2000), 800, 0.2)) == 9
    # tile larger than the region is clamped to the region
    (x, y, w, h), = tiling.plan_tiles((500, 400), None, 800, 0.2)
    assert (x, y, w, h) == (0, 0, 500, 400)


def test_resolve_crop_priority():
    crops = {"default": [0, 0, 10, 10], "landscape": {"crop": [1, 1, 5, 5], "tile": 300},
             "files": {"x.jpg": [2, 2, 6, 6]}}
    assert tiling.resolve_crop("x.jpg", (100, 50), crops, 800) == ((2, 2, 6, 6), 800)
    assert tiling.resolve_crop("y.jpg", (100, 50), crops, 800) == ((1, 1, 5, 5), 300)
    assert tiling.resolve_crop("y.jpg", (50, 100), crops, 800) == ((0, 0, 10, 10), 800)
    assert tiling.resolve_crop("y.jpg", (50, 100), None, 800) == (None, 800)


def test_make_tiles_manifest(photos, tmp_path):
    man = tiling.make_tiles(photos, tmp_path / "t", tmp_path / "m.json", tile=800, overlap=0.2)
    assert set(man["sources"]) == {"a.jpg", "b.jpg", "c.jpg"}
    assert all((tmp_path / "t" / t["file"]).exists() for t in man["tiles"])
    assert json.loads((tmp_path / "m.json").read_text())["params"]["tile"] == 800
    a = [t for t in man["tiles"] if t["source"] == "a.jpg"]
    assert len(a) == 6  # 1600x1200, 800px tiles, 20% overlap -> 3 cols x 2 rows
