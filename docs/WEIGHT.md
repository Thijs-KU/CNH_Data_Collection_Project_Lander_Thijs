# Weight estimation (`detkit/weight/`, `detkit weight ...`)

Photo -> detections -> features -> weight. Two models, both weight = feature x a constant per crop:

| model | feature | constant | in `weight_config.json` |
|---|---|---|---|
| `count_x_constant` (baseline) | `count` (kernels) | grams per kernel | `constants` |
| `area_x_constant` | `total_area_px` (sum of box areas) | grams per px² | `area_constants` |

The area constant depends on the camera distance (pixels per mm), so it holds only for photos taken the same way as
the calibration photos. Which feature to use is task 2 ([STUDENT_TASKS.md](STUDENT_TASKS.md)).

## Files
| file | content | status |
|---|---|---|
| `features.py` | `image_features`, `Sample`, `SampleTable` | works |
| `models.py` | `CountTimesConstant`, `AreaTimesConstant`, `MODELS`, `get_model` | work; bounds `low_g`/`high_g` TODO (task 3) |
| `stats.py` | `evaluate_model`; `learning_curve`, `bootstrap_ci` (task 2); `propagate` (task 3) | last three TODO |
| `config.py` | `load_config`/`save_config` (`<workdir>/weight_config.json`) | works |

## CLI
```bash
python -m detkit weight calibrate --workdir $W --model count_x_constant   # or area_x_constant; fits the constant per crop
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
