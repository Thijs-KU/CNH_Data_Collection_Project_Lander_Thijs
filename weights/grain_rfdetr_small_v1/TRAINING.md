# How this model was trained, and how to reproduce it

Model: `grain_rfdetr_small_v1` (RF-DETR small, 6 classes). Trained **only** on the public *Grain Detection Dataset*
(`coco/train`), model selection on `coco/val`, final report on `coco/test`.
Everything below uses the toolkit in this repository (`detkit`). Labels are automatic and unreviewed, so every score
measures agreement with those labels, not with real grains.

## 1. Environment (versions used)
Python 3.12, PyTorch 2.12.1 (CUDA 12.6 build), transformers 5.12.1, timm 1.0.30, torchmetrics 1.9.0, albumentations 2.0.8,
accelerate 1.15.0, datasets 5.0.1, numpy 2.5.3, Pillow 12.3.0. Install: `uv sync --extra gpu --extra app --extra dev`
(`--extra cpu` on a laptop; see `docs/TRAINING.md`). Do not upgrade `transformers`: the trainer needs the pinned version.

## 2. Data
`datasets/grain_detection_public` (verify: `python scripts/verify.py` prints "all checks passed";
data fingerprint, the sha256 of the checksum lines of `coco/`, `photos/` and `annotations/`: `grep -E "  (coco|photos|annotations)/" checksums.sha256 | sha256sum` = `ed266c44b5444739ddb0cda907e847f37dceff6eb99bb2f8a8ae4bbe3c5e0e2c`; this covers every tile, photo and label used for training and does not change when only the README or scripts are edited).

| split | photos | tiles | boxes |
|---|---|---|---|
| train | 339 | 1,142 | 19,429 |
| val | 65 | 362 | 4,000 |
| test | 66 | 371 | 4,182 |

Tiles are at most 1500 px with 0.2 overlap; classes (ids 0-5): corn, peanuts, popcorn_corn, pumpkin, sunflower, wheat.
The test split is **not** linked into the training workdir, so it cannot be used by accident.

## 3. Train (about 2 hours on one 8 GB GPU)
```bash
cd data_collection_project
source .venv/bin/activate      # once in every new terminal (see README)
python -m detkit init --workdir runs/repro --flow gpu --tile 1500 --overlap 0.2 \
        --classes corn,peanuts,popcorn_corn,pumpkin,sunflower,wheat
mkdir -p runs/repro/coco
ln -s "$(realpath ../datasets/grain_detection_public/coco/train)" runs/repro/coco/train
ln -s "$(realpath ../datasets/grain_detection_public/coco/val)"   runs/repro/coco/val

PYTHONHASHSEED=1337 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m detkit train --workdir runs/repro --name aug --model small --epochs 30 \
         --batch-size 2 --grad-accum 8 --lr 1e-4 --weight-decay 1e-4 \
         --seed 1337 --num-workers 4 --eval-batch-size 4 --logging-steps 10
# result: runs/repro/checkpoints/aug/final   (best validation checkpoint)
```
`--batch-size 2 --grad-accum 8` is an effective batch of 16 that fits an 8 GB GPU (batch 4 and 8 ran out of memory on dense
tiles). With a 24 GB+ GPU use `--batch-size 16 --grad-accum 1` (same effective batch). The same command also works with
`--flow hf` and on CPU (`--device cpu`, but 30 epochs take days: use a GPU or Colab; see `docs/TRAINING.md`).
Comparison run without augmentation: same command with `--name noaug --no-augment`.

| setting | value |
|---|---|
| model | Roboflow/rf-detr-small, 512x512 input, 300 queries |
| epochs / steps | 30 / 2160 optimizer steps |
| batch | 2 x 8 accumulation = 16 |
| learning rate | 1e-4, cosine schedule, 5 % warm-up (the default 5e-5 was raised; linear scaling to 4e-4 is too aggressive with grad-clip 0.01; not tuned further) |
| optimizer | AdamW (fused), weight decay 1e-4, max grad norm 0.01, bf16 |
| augmentation | Perspective p=0.1, HorizontalFlip p=0.5, RandomBrightnessContrast p=0.5, HueSaturationValue p=0.1 |
| seed | 1337 (also data seed and PYTHONHASHSEED) |
| selection | best validation mAP@[.5:.95] over epochs (here: the last epoch) |

Full machine-readable settings: `training_config/hyperparameters.json`. Per-epoch metrics: `training_config/history_aug.json`
and `history_noaug.json`; trainer output: `training_config/train_aug.log`.

## 4. What the training should look like
Validation mAP@[.5:.95] on `coco/val` tiles (augmentation on): epoch 1: 0.05, 2: 0.65, 3: 0.79, 5: 0.82, 10: 0.86,
15: 0.86, 20: 0.88, 25: 0.88, **30: 0.887**; mAP@0.5 reaches 0.98 by epoch 15. It rises fast for 3 epochs, then slowly;
the last 5 epochs change by less than 0.003. About 4 minutes per epoch on a GTX 1070 Ti (about 2 h 5 min total).
If epoch 3 is still below ~0.7, check that `coco/train` is linked correctly and that the learning rate is 1e-4.

## 5. Evaluate (threshold chosen on val, then fixed for test)
```bash
DS=../datasets/grain_detection_public ; CK=runs/repro/checkpoints/aug/final
# (a) val, photo level: tiled inference on full photos; take the best-F1 score from the report (expect ~0.38)
python -m detkit eval --workdir runs/repro --gt $DS/annotations/val_photos.json --images $DS/photos/val --level source \
        --tile 1500 --overlap 0.2 --model $CK --score 0.05 --map --op-score 0.5 --out runs/repro/eval/val_photo --name val_photo \
        --gt-note "automatic labels"
# (b) re-score val and test at that threshold (here 0.384)
TH=0.384
for S in val test; do
  python -m detkit eval --workdir runs/repro --gt $DS/annotations/${S}_photos.json --images $DS/photos/$S --level source \
          --tile 1500 --overlap 0.2 --model $CK --score 0.05 --map --op-score $TH --out runs/repro/eval/${S}_photo --name ${S}_photo \
          --gt-note "automatic labels"
  python -m detkit eval --workdir runs/repro --gt $DS/coco/$S/labels.json --images $DS/coco/$S/images --level tile \
          --model $CK --score 0.05 --map --op-score $TH --out runs/repro/eval/${S}_tile --name ${S}_tile --gt-note "automatic labels"
done
```
Never choose the threshold on test. Reports (`eval.md`, `per_class.csv`, plots) are written to `runs/repro/eval/...`.

## 6. Expected results (what this checkpoint scored, threshold 0.384)
| split | level | precision | recall | F1 | AP@0.5 | mAP@[.5:.95] | count MAE / MAPE |
|---|---|---|---|---|---|---|---|
| val | photo | 0.988 | 0.983 | 0.986 | 0.991 | 0.893 | 0.62 / 2.8 % |
| test | photo | 0.975 | 0.968 | 0.972 | 0.972 | 0.842 | 1.15 / 3.2 % |
| val | tile | 0.943 | 0.985 | 0.964 | 0.991 | 0.892 | 0.68 / 12.0 % |
| test | tile | 0.943 | 0.970 | 0.956 | 0.975 | 0.846 | 0.79 / 14.9 % |

Per class on test photos (F1): corn 0.987, peanuts 0.980, popcorn_corn 0.993, **pumpkin 0.918**, sunflower 0.968, wheat 0.983.
Without augmentation (same settings) validation mAP is lower: 0.870 vs 0.887 (tile), 0.877 vs 0.893 (photo), and the best
score threshold moves to about 0.16-0.29 (the no-augmentation model was only evaluated on val).

**Reproducibility tolerance:** seeds are fixed but CUDA/cuDNN kernels are not forced deterministic, so a re-run is not
bit-identical. Expect validation mAP within about ±0.005 and F1 within about ±0.005; a checkpoint far outside that means a
setup difference (data link, versions, learning rate, effective batch). Test scores are noisy by design: 66 photos from a
handful of independent groups per class.

## 7. Known problems seen in this run
* Pumpkin is weakest; one test photo has labels that look like a broken grid and costs ~36 errors by itself.
* Boxes are a little loose: mAP drops to 0.68 at IoU 0.9.
* Tile-level count error is inflated by grains cut at tile borders; use photo-level numbers for counting.
* Scores are not calibrated probabilities. Use the val-chosen threshold (0.38), not 0.5.
