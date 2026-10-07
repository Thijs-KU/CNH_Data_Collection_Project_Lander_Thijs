import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def photos(tmp_path):
    """3 synthetic photos: 2 landscape (1600x1200), 1 portrait (1000x1400)."""
    d = tmp_path / "photos"
    d.mkdir()
    for name, size in [("a.jpg", (1600, 1200)), ("b.jpg", (1600, 1200)), ("c.jpg", (1000, 1400))]:
        im = Image.new("RGB", size, (200, 200, 200))
        ImageDraw.Draw(im).ellipse([100, 100, 160, 160], fill=(120, 80, 30))
        im.save(d / name)
    return d