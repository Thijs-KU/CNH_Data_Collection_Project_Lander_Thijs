# Evaluating the detector (`detkit eval`)

```bash
# val tiles from `detkit split`
python -m detkit eval --workdir $W --gt val --model $W/checkpoints/run1/final --score 0.05
# whole reviewed photos (tiled inference)
python -m detkit eval --workdir $W --gt reviewed --model $W/checkpoints/run1/final --score 0.05 --images $W/photos
# any COCO file / saved predictions
python -m detkit eval --workdir $W --gt path/to/labels.json --pred predictions.json
```
Useful flags: `--map` (mAP@[.5:.95]), `--iou`, `--crop`, `--name`, `--gt-note "human-reviewed val"` (records where the
ground truth came from).

## Outputs
Written to `<workdir>/reports/eval_<name>/`:
* `eval.md`: readable report (precision, recall, F1, AP, best-F1 score threshold, count and area error).
* CSV files per photo, including `matched_boxes.csv` (predicted/true area ratio per matched box).
* PNG plots.

Use the **best-F1 score** from the report as `--score` in `predict` and in the app. State in every report whether the
ground truth was human-reviewed or pseudo-labels.
