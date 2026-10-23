# Detection evaluation: predict_defaults

- GT: `runs\task2\annotations\reviewed_coco.json` (source-level, 63 images, 4587 GT boxes)
- Predictions: file `runs\task2\predictions\predictions.json`
- IoU threshold 0.5, operating score 0.2
- **GT caveat:** human-reviewed boxes; predictions at score 0.2, tile 1000


## Headline

| | precision | recall | F1 | AP@IoU | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| micro @ score 0.2 | 1.000 | 0.995 | 0.997 | 0.995 | 4564 | 0 | 23 |
| best-F1 @ score 0.202 | 1.000 | 0.995 | 0.997 | - | 4564 | 0 | 23 |
| macro over classes @ 0.2 | 1.000 | 0.995 | 0.997 | 0.995 | | | |

**Recommended operating score: 0.202** (maximises micro F1 on this GT; re-tune on your own reviewed data before trusting it).

ROC-style AUC over predictions = n/a. AUC that a random true detection outranks a random false one by confidence; no true negatives exist in detection, so this is not 1-specificity.

## Per class

| class | GT | precision | recall | F1 | AP | best-F1 score |
|---|---|---|---|---|---|---|
| corn | 0 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| peanuts | 0 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| popcorn_corn | 4587 | 1.000 | 0.995 | 0.997 | 0.995 | 0.202 |
| pumpkin | 0 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| sunflower | 0 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| wheat | 0 | 0.000 | 0.000 | 0.000 | n/a | n/a |

## Count and area error (operating score 0.2)

Per-image totals over all classes; bias = mean(pred - GT).

| quantity | n images | bias | MAE | RMSE | MAPE % | R2 | total error % |
|---|---|---|---|---|---|---|---|
| count | 63 | -0.37 | 0.37 | 1.03 | 0.4 | 0.994 | -0.5 |
| box area px^2 | 63 | -721 | 722 | 2001 | 0.5 | 0.993 | -0.5 |

Worst images by |count error|: Sample18_0.jpeg (GT 100, pred 95, -5%); Sample18_1.jpeg (GT 100, pred 96, -4%); Sample18_2.jpeg (GT 100, pred 96, -4%); Sample14_0.jpeg (GT 90, pred 89, -1%); Sample14_2.jpeg (GT 90, pred 89, -1%)


## Count and area error (best-F1 score 0.202)

Per-image totals over all classes; bias = mean(pred - GT).

| quantity | n images | bias | MAE | RMSE | MAPE % | R2 | total error % |
|---|---|---|---|---|---|---|---|
| count | 63 | -0.37 | 0.37 | 1.03 | 0.4 | 0.994 | -0.5 |
| box area px^2 | 63 | -721 | 722 | 2001 | 0.5 | 0.993 | -0.5 |

Worst images by |count error|: Sample18_0.jpeg (GT 100, pred 95, -5%); Sample18_1.jpeg (GT 100, pred 96, -4%); Sample18_2.jpeg (GT 100, pred 96, -4%); Sample14_0.jpeg (GT 90, pred 89, -1%); Sample14_2.jpeg (GT 90, pred 89, -1%)

## Per-photo table (operating score)

| file | GT | pred | TP | FP | FN | count err % | GT area | pred area | area err % |
|---|---|---|---|---|---|---|---|---|---|
| IMG_8854.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 137167 | 137167 | 0.0 |
| IMG_8855.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 137698 | 137698 | 0.0 |
| IMG_8856.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 136833 | 136833 | 0.0 |
| Sample10_0.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 132159 | 132159 | 0.0 |
| Sample10_1.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 131211 | 131211 | -0.0 |
| Sample10_2.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 131177 | 131177 | 0.0 |
| Sample11_0.jpeg | 50 | 50 | 50 | 0 | 0 | 0.0 | 95473 | 95473 | 0.0 |
| Sample11_1.jpeg | 50 | 50 | 50 | 0 | 0 | 0.0 | 95706 | 95457 | -0.3 |
| Sample11_2.jpeg | 50 | 50 | 50 | 0 | 0 | 0.0 | 93642 | 93254 | -0.4 |
| Sample12_0.jpeg | 62 | 62 | 62 | 0 | 0 | 0.0 | 121273 | 121282 | 0.0 |
| Sample12_1.jpeg | 62 | 62 | 62 | 0 | 0 | 0.0 | 120358 | 120358 | 0.0 |
| Sample12_2.jpeg | 62 | 62 | 62 | 0 | 0 | 0.0 | 119996 | 119996 | 0.0 |
| Sample13_0.jpeg | 80 | 80 | 80 | 0 | 0 | 0.0 | 156958 | 156958 | 0.0 |
| Sample13_1.jpeg | 80 | 80 | 80 | 0 | 0 | 0.0 | 156269 | 156269 | 0.0 |
| Sample13_2.jpeg | 80 | 80 | 80 | 0 | 0 | 0.0 | 157149 | 157149 | 0.0 |
| Sample14_0.jpeg | 90 | 89 | 89 | 0 | 1 | -1.1 | 172556 | 169632 | -1.7 |
| Sample14_1.jpeg | 90 | 90 | 90 | 0 | 0 | 0.0 | 172136 | 172136 | 0.0 |
| Sample14_2.jpeg | 90 | 89 | 89 | 0 | 1 | -1.1 | 171508 | 169194 | -1.3 |
| Sample15_0.jpeg | 100 | 100 | 100 | 0 | 0 | 0.0 | 186371 | 186319 | -0.0 |
| Sample15_1.jpeg | 100 | 100 | 100 | 0 | 0 | 0.0 | 186099 | 186099 | 0.0 |
| Sample15_2.jpeg | 100 | 100 | 100 | 0 | 0 | 0.0 | 186195 | 186195 | -0.0 |
| Sample16_0.jpeg | 50 | 50 | 50 | 0 | 0 | 0.0 | 94785 | 94785 | -0.0 |
| Sample16_1.jpeg | 50 | 50 | 50 | 0 | 0 | 0.0 | 97039 | 97033 | -0.0 |
| Sample16_2.jpeg | 50 | 50 | 50 | 0 | 0 | 0.0 | 96120 | 96120 | -0.0 |
| Sample17_0.jpeg | 60 | 60 | 60 | 0 | 0 | 0.0 | 128963 | 128958 | -0.0 |
| Sample17_1.jpeg | 60 | 59 | 59 | 0 | 1 | -1.7 | 126905 | 124853 | -1.6 |
| Sample17_2.jpeg | 60 | 59 | 59 | 0 | 1 | -1.7 | 118816 | 116171 | -2.2 |
| Sample18_0.jpeg | 100 | 95 | 95 | 0 | 5 | -5.0 | 178912 | 170405 | -4.8 |
| Sample18_1.jpeg | 100 | 96 | 96 | 0 | 4 | -4.0 | 180367 | 173223 | -4.0 |
| Sample18_2.jpeg | 100 | 96 | 96 | 0 | 4 | -4.0 | 179228 | 169954 | -5.2 |
| Sample19_0.jpeg | 80 | 80 | 80 | 0 | 0 | 0.0 | 150886 | 150886 | 0.0 |
| Sample19_1.jpeg | 80 | 80 | 80 | 0 | 0 | 0.0 | 150274 | 150274 | 0.0 |
| Sample19_2.jpeg | 80 | 80 | 80 | 0 | 0 | 0.0 | 151156 | 151156 | -0.0 |
| Sample1_0.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 137547 | 137547 | 0.0 |
| Sample1_1.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 136817 | 136817 | 0.0 |
| Sample1_2.jpeg | 70 | 69 | 69 | 0 | 1 | -1.4 | 136539 | 135077 | -1.1 |
| Sample20_0.jpeg | 90 | 90 | 90 | 0 | 0 | 0.0 | 168814 | 168814 | 0.0 |
| Sample20_1.jpeg | 90 | 90 | 90 | 0 | 0 | 0.0 | 164431 | 164431 | 0.0 |
| Sample20_2.jpeg | 90 | 89 | 89 | 0 | 1 | -1.1 | 165938 | 163715 | -1.3 |
| Sample2_0.jpeg | 69 | 69 | 69 | 0 | 0 | 0.0 | 127518 | 127522 | 0.0 |
| Sample2_1.jpeg | 69 | 69 | 69 | 0 | 0 | 0.0 | 127082 | 127082 | 0.0 |
| Sample2_2.jpeg | 69 | 69 | 69 | 0 | 0 | 0.0 | 126806 | 126806 | 0.0 |
| Sample3_0.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 133021 | 133021 | -0.0 |
| Sample3_1.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 132779 | 132779 | -0.0 |
| Sample3_2.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 134309 | 134309 | -0.0 |
| Sample4_0.jpeg | 68 | 68 | 68 | 0 | 0 | 0.0 | 124753 | 124757 | 0.0 |
| Sample4_1.jpeg | 68 | 68 | 68 | 0 | 0 | 0.0 | 124072 | 124072 | 0.0 |
| Sample4_2.jpeg | 68 | 68 | 68 | 0 | 0 | 0.0 | 123167 | 123167 | -0.0 |
| Sample5_0.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 128940 | 128935 | -0.0 |
| Sample5_1.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 129585 | 129585 | 0.0 |
| Sample5_2.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 129552 | 129552 | 0.0 |
| Sample6_0.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 132172 | 132172 | 0.0 |
| Sample6_1.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 129840 | 129840 | 0.0 |
| Sample6_2.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 131011 | 131011 | 0.0 |
| Sample7_0.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 128144 | 128144 | -0.0 |
| Sample7_1.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 127745 | 127745 | -0.0 |
| Sample7_2.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 129059 | 129059 | -0.0 |
| Sample8_0.jpeg | 70 | 69 | 69 | 0 | 1 | -1.4 | 129375 | 128099 | -1.0 |
| Sample8_1.jpeg | 70 | 69 | 69 | 0 | 1 | -1.4 | 128942 | 127802 | -0.9 |
| Sample8_2.jpeg | 70 | 69 | 69 | 0 | 1 | -1.4 | 128744 | 127223 | -1.2 |
| Sample9_0.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 125734 | 125734 | 0.0 |
| Sample9_1.jpeg | 70 | 69 | 69 | 0 | 1 | -1.4 | 125424 | 123156 | -1.8 |
| Sample9_2.jpeg | 70 | 70 | 70 | 0 | 0 | 0.0 | 125247 | 125247 | 0.0 |

## Matched boxes (score >= operating score)

4564 matched pairs; mean IoU 1.000; pred/GT area ratio median 1.000, mean 1.000, p10-p90 [1.0, 1.0]. A ratio far from 1 means the boxes are systematically too large/small, which biases pixels -> weight even when the count is right. Full table: `matched_boxes.csv`.

## Files

`eval.json`, `per_image.csv`, `per_class.csv`, `threshold_sweep.csv`, `pr_curve.csv`, `matched_boxes.csv`, `pr_curve.png`, `f1_vs_threshold.png`, `roc.png`, `count_scatter.png`, `area_scatter.png`.

## Kernel-pixel area error (score >= 0.2)

Kernel pixels inside the boxes (`weight/features.py: kernel_mask`), predicted boxes against GT boxes, per photo.

| n photos | bias px^2 | MAE px^2 | RMSE px^2 | MAPE % | R2 |
|---|---|---|---|---|---|
| 63 | -497 | 497 | 1379 | 0.4 | 0.994 |

Files: `kernel_area.csv`, `kernel_area_scatter.png`.
