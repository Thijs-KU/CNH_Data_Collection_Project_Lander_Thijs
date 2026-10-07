"""Project config (``<workdir>/project.json``) and the standard directory layout.

Every stage reads its defaults from here so a new batch of images with different
classes is just ``detkit init --workdir runs/x --classes a,b`` followed by the stages.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


# ---- flows -------------------------------------------------------------------------------
# A flow is a preset of *where each step runs*. Same commands in all three.
#   hf  : training runs on this machine (GPU if there is one); use with --push-to-hub to share a model.
#   cpu : no GPU. CPU-friendly training defaults (nano, batch 2, no dataloader workers).
#   gpu : local GPU.
FLOWS: dict[str, dict] = {
    "hf": dict(device="auto", train_model="small", train_batch=4, train_workers=2),
    "cpu": dict(device="cpu", train_model="nano", train_batch=2, train_workers=0),
    "gpu": dict(device="cuda", train_model="small", train_batch=4, train_workers=2),
}


@dataclass
class Project:
    workdir: Path
    classes: list[str] = field(default_factory=lambda: ["seed"])
    tile: int = 800
    overlap: float = 0.2
    crops: str | None = None                  # path to a crops.json (see tiling.py)
    # flow-driven settings (see FLOWS); `detkit init --flow` fills them
    flow: str = "hf"
    device: str = "auto"                       # auto | cpu | cuda
    proposer_ckpt: str | None = None           # RF-DETR checkpoint dir (e.g. weights/grain_rfdetr_small_v1/final)
    train_model: str = "small"
    train_batch: int = 4
    train_workers: int = 2
    extra_image_fields: list[dict] = field(default_factory=list)   # per-photo properties added in the app

    # ---- layout -------------------------------------------------------
    @property
    def tiles_dir(self) -> Path: return self.workdir / "tiles"
    @property
    def manifest(self) -> Path: return self.workdir / "tiles_manifest.json"
    @property
    def ann_dir(self) -> Path: return self.workdir / "annotations"
    @property
    def tiles_coco(self) -> Path: return self.ann_dir / "tiles_coco.json"
    @property
    def viz_dir(self) -> Path: return self.workdir / "viz"
    @property
    def coco_dir(self) -> Path: return self.workdir / "coco"
    @property
    def ckpt_dir(self) -> Path: return self.workdir / "checkpoints"
    @property
    def photos_dir(self) -> Path: return self.workdir / "photos"
    @property
    def reviewed_coco(self) -> Path: return self.ann_dir / "reviewed_coco.json"
    @property
    def image_meta(self) -> Path: return self.workdir / "meta" / "image_meta.json"
    @property
    def reports_dir(self) -> Path: return self.workdir / "reports"

    def image_fields(self):
        from .schema import image_fields
        return image_fields(self.extra_image_fields)

    def store(self):
        from .store import Store
        return Store(self.workdir, self.classes, [e["key"] for e in self.extra_image_fields])

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    # ---- persistence ---------------------------------------------------
    @property
    def path(self) -> Path: return self.workdir / "project.json"

    def save(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d.pop("workdir")
        self.path.write_text(json.dumps(d, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, workdir: str | Path) -> "Project":
        workdir = Path(workdir)
        p = workdir / "project.json"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run: detkit init --workdir {workdir} --classes a,b")
        data = json.loads(p.read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}      # tolerate old/new project.json versions
        return cls(workdir=workdir, **{k: v for k, v in data.items() if k in known})


def parse_classes(text: str | list[str]) -> list[str]:
    items = text.split(",") if isinstance(text, str) else text
    out = [c.strip() for c in items if c.strip()]
    if not out:
        raise ValueError("at least one class is required")
    if len({c.lower() for c in out}) != len(out):
        raise ValueError(f"duplicate classes in {out}")
    return out


def load_env(start: Path | None = None, levels: int = 4) -> Path | None:
    """Load KEY=VALUE lines from the nearest ``.env`` (cwd and up to *levels* parents).

    Existing environment variables win. Values are never printed. Returns the file used.
    """
    d = (start or Path.cwd()).resolve()
    for p in [d, *list(d.parents)[:levels]]:
        f = p / ".env"
        if f.is_file():
            for line in f.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            return f
    return None
