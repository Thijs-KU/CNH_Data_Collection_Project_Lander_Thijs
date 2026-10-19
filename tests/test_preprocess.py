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
