"""Preprocessing hook: a no-op until configured; geometry only changes through the canonicalizer."""

import numpy as np
import pytest
from PIL import Image

from detkit.preprocess import Preprocessor


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "a.png"
    Image.fromarray((np.random.rand(40, 60, 3) * 255).astype("uint8")).save(p)
    return p


def test_default_is_unchanged(photo, tmp_path):
    pre = Preprocessor()
    im = Image.open(photo)
    assert pre.apply(im) is im and pre.canonicalize(im) is im
    assert pre.canonical_path(photo, tmp_path / "c") == photo
    dst = pre.ingest(photo, tmp_path / "b.png")
    assert dst.read_bytes() == photo.read_bytes()


def test_apply_steps_keep_size(photo):
    im = Image.open(photo).convert("RGB")
    out = Preprocessor(steps=[lambda b: 255 - b]).apply(im)
    assert out.size == im.size and out is not im
    with pytest.raises(ValueError):
        Preprocessor(steps=[lambda b: b[:10]]).apply(im)


def test_canonicalizer_changes_stored_photo(photo, tmp_path):
    pre = Preprocessor(canonicalizer=lambda im: im.resize((30, 20)))
    assert Image.open(pre.ingest(photo, tmp_path / "b.png")).size == (30, 20)
    cp = pre.canonical_path(photo, tmp_path / "cache")
    assert cp.parent == tmp_path / "cache" and cp.name == photo.name and Image.open(cp).size == (30, 20)


def test_eval_frame_guard():
    from detkit.cmd_eval import _check_frame
    _check_frame("a.jpg", (60, 40), {"width": 60, "height": 40})
    _check_frame("a.jpg", (60, 40), {})                                  # no size recorded: nothing to check
    with pytest.raises(ValueError, match="canonical"):
        _check_frame("a.jpg", (60, 40), {"width": 30, "height": 20})


def test_cli_flags():
    from detkit.cli import build_parser
    ap = build_parser()
    assert ap.parse_args(["predict", "--model", "m", "--images", "x", "--canonicalize"]).canonicalize
    assert not ap.parse_args(["predict", "--model", "m", "--images", "x"]).canonicalize
    assert ap.parse_args(["eval", "--gt", "reviewed", "--model", "m", "--canonicalize"]).canonicalize


# ---- brightness normalization outside the project (predict / eval / Inference tab) ---------------------------
def test_normalize_defaults_and_cli_flags():
    from detkit.cli import build_parser
    from detkit.preprocess import DEFAULT_CONTRAST, TARGET_BRIGHTNESS
    assert (TARGET_BRIGHTNESS, DEFAULT_CONTRAST) == (220, 1.0)
    ap = build_parser()
    a = ap.parse_args(["predict", "--model", "m", "--images", "x"])
    assert not a.normalize and a.target_brightness == 220 and a.contrast == 1.0
    a = ap.parse_args(["eval", "--gt", "reviewed", "--model", "m", "--normalize", "--target-brightness", "200",
                       "--contrast", "0.5"])
    assert a.normalize and a.target_brightness == 200 and a.contrast == 0.5


def test_prepare_photo_skips_project_photos(tmp_path):
    from detkit.preprocess import before_after, brightness, in_project_photos, prepare_photo
    im = Image.new("RGB", (60, 40), (130, 130, 130))
    raw = tmp_path / "raw" / "a.png"
    proj = tmp_path / "w"
    (proj / "photos").mkdir(parents=True)
    (proj / "project.json").write_text("{}")
    assert not in_project_photos(raw) and in_project_photos(proj / "photos" / "a.png")
    out, info = prepare_photo(im, raw, (220, 1.0))
    assert info and abs(brightness(out) - 220) <= 2 and out.size == im.size
    same, none = prepare_photo(im, proj / "photos" / "a.png", (220, 1.0))
    assert same is im and none is None
    assert prepare_photo(im, raw, None) == (im, None)
    pic = before_after(im, out, info, "a.png")
    assert pic.height > 40 and pic.width > 60
