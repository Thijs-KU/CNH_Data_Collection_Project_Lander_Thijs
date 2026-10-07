# Grain detector (RF-DETR small), v1

An object detector that finds and classifies small grains and seeds lying on a flat surface: **corn, peanuts,
popcorn_corn, pumpkin, sunflower, wheat**. Each photo is expected to contain one kind of grain; the model draws one
box per grain and names its class. It is a fine-tuned **RF-DETR small** (about 32 M parameters) and was trained for a
course on object detection.

**Intended use:** education and non-commercial experiments (counting grains, estimating pile area or weight from
detections, comparing evaluation settings). License: **CC BY-NC 4.0**, following the training dataset.
Not for commercial use and not for any safety-relevant decision.

**Links:** dataset https://huggingface.co/datasets/niru-5/manama_course_dataset_grain_detection | model https://huggingface.co/niru-5/manama_course_rfdetr_grain_detection | code https://github.com/niru-5/manama_data_collection_course

## Training data
The public *Grain Detection Dataset (course release)*: 470 photos cut into tiles of at most 1500 px with 20 % overlap
(1,875 tiles, 27,611 boxes, six classes). This model was trained on the train split only (339 photos, 1,142 tiles,
19,429 boxes) and selected on the validation split (65 photos, 362 tiles). The test split (66 photos) was used only for
the final report. The splits are by photo group, never by tile.

**Honesty caveat: the dataset labels are automatic labels that have not been reviewed by a human.**
All numbers below measure agreement with those labels, not with the real grains. Real accuracy can be lower or higher.

## Training settings (short)
RF-DETR small, 6 classes, tile-level training, 30 epochs, effective batch size 16 (2 x 8 gradient accumulation),
learning rate 1e-4 (cosine, 5 % warm-up), weight decay 1e-4, AdamW, images resized to 512 x 512 by the model's
processor, augmentation on (horizontal flip, brightness/contrast, small colour and perspective changes), seed 1337.
The checkpoint with the best validation mAP is shipped (the last epoch). A run without augmentation reached a lower
validation mAP (0.870 vs 0.887 tile-level; 0.877 vs 0.893 photo-level).

**Full training log and step-by-step reproduction (commands, versions, expected curve and scores): see `TRAINING.md`.**

## Recommended score threshold: **0.38**
Chosen on the validation photos (best F1 at IoU 0.5, tiled inference). Do not use 0.5 by default: the model's scores
are not calibrated probabilities. On the test photos the F1-optimal threshold would have been 0.49, so anything in
0.35 to 0.5 behaves similarly (test F1 0.972 at 0.38 versus 0.974 at 0.49).

## Results (IoU 0.5 unless noted; threshold 0.38, chosen on validation only)
Photo level = full photos, tiled inference with tile 1500 and overlap 0.2. Tile level = the 1500 px tiles one by one.

| split | level | precision | recall | F1 | AP@0.5 | mAP@[.5:.95] | count MAE / MAPE |
|---|---|---|---|---|---|---|---|
| val (65 photos) | photo | 0.988 | 0.983 | 0.986 | 0.991 | 0.893 | 0.62 / 2.8 % |
| test (66 photos) | photo | 0.975 | 0.968 | 0.972 | 0.972 | 0.842 | 1.15 / 3.2 % |
| val (362 tiles) | tile | 0.943 | 0.985 | 0.964 | 0.991 | 0.892 | 0.68 / 12.0 % |
| test (371 tiles) | tile | 0.943 | 0.970 | 0.956 | 0.975 | 0.846 | 0.79 / 14.9 % |

Count error per photo: the summed count over the photo is off by only -0.7 % on test (-0.6 % on val) in total; the
total box area is 2.8 % above the labels on test photos. The tile-level count percentage is large because many tiles
hold few grains and tile borders cut grains (a grain cut by a border is counted differently by labels and model).

Per class, test photos (threshold 0.38):

| class | boxes | precision | recall | F1 | AP@0.5 |
|---|---|---|---|---|---|
| corn | 297 | 0.987 | 0.987 | 0.987 | 0.987 |
| peanuts | 344 | 0.982 | 0.977 | 0.980 | 0.980 |
| popcorn_corn | 480 | 0.994 | 0.992 | 0.993 | 0.994 |
| pumpkin | 432 | 0.917 | 0.919 | 0.918 | 0.922 |
| sunflower | 380 | 0.992 | 0.945 | 0.968 | 0.956 |
| wheat | 695 | 0.981 | 0.984 | 0.983 | 0.987 |

Validation and test are small (about 65 photos each, a handful of independent groups per class), so per-class numbers
are noisy: a few photos can move a class by several points.

## Where it is weakest
- **Pumpkin seeds** (lowest F1, 0.92 on test): peeled, pale kernels are sometimes called corn (seen in at least one test photo), and
  one test photo has labels that look wrong (tiny boxes in a regular grid), which alone costs recall and precision.
- **Sunflower seeds** (recall 0.945): touching and overlapping seeds in dense piles are merged or missed.
- **Tight boxes**: AP falls from 0.97 at IoU 0.5 to 0.68 at IoU 0.9 (test), so box edges are good but not precise;
  area estimates from boxes carry a few percent error.
- **Other backgrounds, lighting or camera distances**: the dataset covers a few backgrounds and one lighting style, so
  expect worse results elsewhere. The model assumes one grain type per photo and does not separate mixed piles
  reliably.

## How to use
With the course toolkit (`detkit`), use the same tiling as in training (stored in `final/detkit_meta.json`):
```
python -m detkit predict --model final --images my_photo.jpg --out predictions --score 0.38 --tile 1500 --overlap 0.2
python -m detkit eval --workdir <your_workdir> --gt <coco.json> --model final --score 0.05 --level source --tile 1500 --overlap 0.2 --map --op-score 0.38
python -m detkit app --workdir <your_workdir> --ckpt final        # use it to propose boxes in the annotation app
```
`final/` holds `config.json`, `model.safetensors`, `preprocessor_config.json` and `detkit_meta.json`; it also loads with
`transformers` (`AutoModelForObjectDetection`, `AutoImageProcessor`). Sample overlays of test photos are in
`eval/overlays_test/`; full evaluation reports are in `eval/`.

## Limitations and responsible use
Trained and evaluated on automatically labelled data of limited variety (see above). Check overlays on your own
photos before trusting counts, and re-tune the threshold on photos you reviewed yourself. The weights inherit the
dataset's non-commercial license.
