# Weight estimation (`detkit/weight/`, `detkit weight ...`)

Photo -> detections -> features -> weight. Three models, all weight = feature x a constant per crop:

| model | feature | constant | in `weight_config.json` |
|---|---|---|---|
| `count_x_constant` (baseline) | `count` (kernels) | grams per kernel | `constants` |
| `area_x_constant` | `total_area_px` (sum of box areas) | grams per px² | `area_constants` |
| `kernel_area_x_constant` | `kernel_area_px` (kernel pixels only, box background removed) | grams per px² | `kernel_area_constants` |

The area constant depends on the camera distance (pixels per mm), so it holds only for photos taken the same way as
the calibration photos. Which feature to use is task 2 ([STUDENT_TASKS.md](STUDENT_TASKS.md)).

## Files
| file | content | status |
|---|---|---|
| `features.py` | `image_features`, `Sample`, `SampleTable` | works |
| `models.py` | `CountTimesConstant`, `AreaTimesConstant`, `MODELS`, `get_model` | work; bounds `low_g`/`high_g` TODO (task 3) |
| `stats.py` | `evaluate_model`; `feature_correlations`, `bootstrap_ci`, `learning_curve` (task 2); `propagate` (task 3) | `propagate` TODO |
| `config.py` | `load_config`/`save_config` (`<workdir>/weight_config.json`) | works |

## CLI
```bash
python -m detkit weight calibrate --workdir $W --model count_x_constant   # or area_x_constant | kernel_area_x_constant; fits the constant per crop
python -m detkit weight features  --workdir $W [--crop C] [--boot 1000] [--plot features.png]   # r of every feature vs weight
python -m detkit weight predict   --workdir $W (--predictions P | --use reviewed) [--model NAME] [--reviewed-only] [--out CSV]
```
`calibrate` saves the constants and makes that model the default. `predict` prints per-photo count, estimated
weight and the error vs `total_weight_g`.

## Python
```python
from detkit.project import Project
from detkit.weight import SampleTable, CountTimesConstant, evaluate_model
t = SampleTable.from_project(Project.load("runs/mine")).with_weight()
m = CountTimesConstant().fit(t)
print(m.constants, evaluate_model(m, t)["mape"])   # error on the photos it was fitted on: optimistic
```

## Kernel-pixel features (`kernel_mask`, `image_features(..., image=)`)
A box is always bigger than the kernel in it (tilt, corners). With the preprocessed photo the features count only the
pixels that are kernel: each pixel gets its colour distance to the paper (median Lab colour outside the boxes), Otsu's
threshold over the pixels in and around the boxes splits paper from kernel, and per box the mask is closed, reduced
to its largest blob and filled to its convex hull (recovers pale kernel tips). Overlapping boxes count a pixel once.
Added keys: `kernel_area_px`, `mean_kernel_area_px`, `median_kernel_area_px`, `kernel_fill_ratio` (kernel / box
pixels; about 0.7 on the sample photos) and `kernel_area_mm2`, `mean_kernel_area_mm2` with a scale.
`SampleTable.from_project` measures them on `<workdir>/photos/<file>` (the preprocessed photo the boxes belong to;
`kernel_area=False` skips it); the Inference tab passes the prepared photo.
