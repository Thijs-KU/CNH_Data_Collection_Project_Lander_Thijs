"""Per-workdir weight config (``W/weight_config.json``): which model, and its constants per crop."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_NAME = "weight_config.json"

# Starting values only: calibrate your own with `detkit weight calibrate` (task 2).
#   constants       grams per kernel      (count_x_constant)
#   area_constants  grams per px^2 of box (area_x_constant; depends on camera distance, so no default)
DEFAULT_CONFIG: dict = {
    "model": "count_x_constant",
    "constants": {"default": 0.04, "wheat": 0.04, "sunflower": 0.05},
    "area_constants": {},
}


def load_config(workdir) -> dict:
    """Config of a workdir; missing file (or missing keys) -> the defaults above."""
    p = Path(workdir) / CONFIG_NAME
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if p.exists():
        saved = json.loads(p.read_text(encoding="utf-8"))
        for k, v in saved.items():
            cfg[k] = {**cfg[k], **v} if isinstance(v, dict) and isinstance(cfg.get(k), dict) else v
    return cfg


def save_config(workdir, cfg: dict) -> Path:
    p = Path(workdir) / CONFIG_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return p
