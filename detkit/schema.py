"""Metadata schema: the single source of truth for what the annotation app asks for and what the
CSV/analysis tables contain. Add a field here and the app form picks it up. ``kind`` is one of: str, float, int, choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Field:
    key: str
    kind: str = "str"
    unit: str = ""
    help: str = ""
    choices: tuple[str, ...] = ()


# Per photo. `crop` doubles as the detector class when the project uses one class per crop.
IMAGE_FIELDS: list[Field] = [
    Field("crop", "str", help="Crop / variety in the photo (also selects the detector class), e.g. wheat"),
    Field("pile_id", "str", help="Same physical pile/sample -> same id (several photos of ONE pile are not independent)"),
    Field("total_weight_g", "float", "g", "Weight of ALL grains in the photo (balance reading)"),
    Field("manual_count", "int", "", "Grains counted by hand (ground truth for count)"),
]
# Grain moisture is assumed constant across all samples, so it is not recorded.

# Per bounding box.
BOX_FIELDS: list[Field] = [
    Field("weight_g", "float", "g", "Weight of this single grain, if it was weighed individually"),
    Field("box_notes", "str"),
]

# Filled automatically (not typed by students); see store.box_origin values.
AUTO_BOX_KEYS = ("origin", "score")     # origin: rfdetr | manual
AUTO_IMAGE_KEYS = ("reviewed", "reviewed_at", "proposer")

IMAGE_KEYS = [f.key for f in IMAGE_FIELDS]
BOX_KEYS = [f.key for f in BOX_FIELDS]

# Extra per-photo properties a project adds itself (app: "Add an extra property"; stored in
# project.json -> extra_image_fields as [{"key", "kind", "unit"}]), e.g. annotator, moisture_pct.
EXTRA_KINDS = ("str", "float", "int")


def image_fields(extra: list[dict] | None = None) -> list[Field]:
    """IMAGE_FIELDS followed by the project's extra properties."""
    return IMAGE_FIELDS + [Field(e["key"], e.get("kind", "str"), e.get("unit", "")) for e in extra or []]


def coerce(f: Field, value):
    """Convert a UI/CSV value to the field's type; empty -> None."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if f.kind == "float":
        return float(value)
    if f.kind == "int":
        return int(float(value))
    return str(value).strip()
