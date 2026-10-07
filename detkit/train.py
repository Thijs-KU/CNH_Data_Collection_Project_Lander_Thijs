"""Train a small RF-DETR on ``<workdir>/coco`` (built by ``detkit split``)."""

from __future__ import annotations

import json
from pathlib import Path

MODELS = {
    "nano": "Roboflow/rf-detr-nano",
    "small": "Roboflow/rf-detr-small",
    "base": "Roboflow/rf-detr-base",
    "medium": "Roboflow/rf-detr-medium",
    "large": "Roboflow/rf-detr-large",
}
META = "detkit_meta.json"


def train_model(coco_dir: str | Path, out_dir: str | Path, *, model: str = "small",
                epochs: int = 30, batch_size: int = 4, lr: float = 5e-5, grad_accum: int = 1,
                augment: bool = True, num_workers: int = 2, logging_steps: int = 5,
                load_best: bool = True, tile_params: dict | None = None,
                report_to: str = "none", push_to_hub: bool = False,
                hub_model_id: str | None = None, seed: int = 1337,
                eval_batch_size: int | None = None, weight_decay: float = 1e-4) -> Path:
    from .vendor.train_rfdetr import train

    coco_dir, out_dir = Path(coco_dir), Path(out_dir)
    if not (coco_dir / "train" / "labels.json").exists():
        raise FileNotFoundError(f"{coco_dir}/train/labels.json missing - run `detkit split` first")
    has_val = (coco_dir / "val" / "labels.json").exists()
    model_id = MODELS.get(model, model)
    final = train(
        coco_dir, output_dir=out_dir, model_id=model_id, epochs=epochs, batch_size=batch_size,
        lr=lr, gradient_accumulation_steps=grad_accum, augment=augment,
        train_split="train", val_split="val" if has_val else None, num_workers=num_workers,
        logging_steps=logging_steps, load_best=load_best, report_to=report_to,
        push_to_hub=push_to_hub, hub_model_id=hub_model_id, seed=seed,
        eval_batch_size=eval_batch_size, weight_decay=weight_decay,
    )
    classes = [c["name"] for c in sorted(
        json.loads((coco_dir / "train" / "labels.json").read_text(encoding="utf-8"))["categories"],
        key=lambda c: c["id"])]
    meta = {"model_id": model_id, "classes": classes, "epochs": epochs, "batch_size": batch_size,
            "augment": augment, "seed": seed, "lr": lr, "grad_accum": grad_accum,
            "tile_params": tile_params or {}}
    (Path(final) / META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return Path(final)
