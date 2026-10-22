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
| `features.py` | `image_features`, `kernel_mask`, `Sample`, `SampleTable` | works |
| `models.py` | `CountTimesConstant`, `AreaTimesConstant`, `KernelAreaTimesConstant`, `MODELS`, `get_model` | work; bounds `low_g`/`high_g` TODO (task 3) |
| `stats.py` | `evaluate_model`; `feature_correlations`, `numeric_features`, `bootstrap_ci`, `learning_curve` (task 2); `propagate` (task 3) | `propagate` TODO |
| `plots.py` | the figures of `python -m detkit weight features` (matplotlib) | works |
| `config.py` | `load_config`/`save_config` (`<workdir>/weight_config.json`) | works |

## CLI
```bash
python -m detkit weight calibrate --workdir $W --model count_x_constant   # or area_x_constant | kernel_area_x_constant; fits the constant per crop
python -m detkit weight features  --workdir $W [--crop C] [--boot 1000] [--curve FEATURE] [--plot all.png] [--plot-dir DIR] [--format png|pdf|svg] [--show]
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

## Feature statistics and plots (`python -m detkit weight features`, `stats.py`, `plots.py`)
Which feature follows the measured weight, how sure is that, and how many samples does it take? The statistics are in `stats.py` (`feature_correlations`, `bootstrap_ci`,
`learning_curve`: the task 2 functions, with their original names) and the figures in `plots.py`.
```bash
python -m detkit weight features --workdir $W --boot 1000 --curve kernel_area_px \
    --plot all.png --plot-dir feature_plots --format pdf
```
| option | what it does |
|---|---|
| `--crop C` | only photos of this crop |
| `--boot N` | adds the 95 % bootstrap interval of r (N rounds, 0 = off) |
| `--curve FEATURE` | learning curve of that feature, several comma separated or `all` (`--sizes 5,10,20`, `--repeats 200`); always plotted: in `--plot-dir`, next to `--plot`, else in `<workdir>/reports/features`; with several features also a comparison figure `learning_curve_all` |
| `--plot FILE` | one overview figure with every feature |
| `--plot-dir DIR` | one larger figure per feature (+ `learning_curve_<feature>` with `--curve`) |
| `--format` | file type for `--plot-dir` and the curve: `png` (default), `pdf`, `svg` |
| `--show` | open the figures in windows (needs a display) |

Short version of all options and commands: [CHEATSHEET_FEATURES.md](CHEATSHEET_FEATURES.md).

**Table.** For every numeric feature: r, r², the least-squares slope and (`--boot`) the interval of r, best first.
A second table always follows (also without weights): min, median, mean, p95, max and std of every feature over all
photos.
Features that never change (r undefined) are left out. Only photos with a weight > 0 count; weights <= 0 are
reported and left out. A high r is not enough to choose: count, box area and kernel area all grow with the pile, so
they all score high. Compare the calibration error (`calibrate` + `predict`, on photos not used for calibration) and
look at the intercept: `weight = feature x constant` goes through the origin, r does not check that.

**Piles.** Photos with the same `pile_id` are photos of one physical pile: they have the same weight, so they are not
independent samples. `bootstrap_ci` resamples whole piles (cluster bootstrap) and `learning_curve` draws subsets made
of whole piles; resampling single photos would give intervals that are too narrow. A photo without a `pile_id` is a
pile of its own. Fewer than 3 weighed photos (or, for the bootstrap, fewer than 2 piles) gives `None`.

**Bootstrap (`bootstrap_ci`).** `n_boot` times: draw as many piles as there are, with replacement, recompute r (or the
slope, `stat="slope"`); the interval is the middle 95 % (`alpha`) of those values. Reproducible with `seed`.

**Learning curve (`learning_curve`).** For every size in `sizes` (>= 3 photos; 2 points always give r = +-1): `repeats`
random subsets, piles in random order, the last one cut to the size; fit weight on the feature. Per size the mean,
std and the middle 95 % of r and of the slope. Read the *width* of that range, not r: with few samples r can be
anywhere (3 points can lie on a line by chance), and the curve has settled when the range stops getting narrower.
The number of piles matters more than the number of photos. At the full size every repeat uses all the data, so its
spread is 0 by construction (drawn as a reference line, not as a band).

**Plots.** Every pile has its own colour and marker (the legend lists the piles and their photo count); photos of one
pile sit in one horizontal row and the hollow diamond is the pile mean. The red line is the free least-squares fit, the
dashed line is the line through the origin that the weight models use (its slope is the constant). Without
weights (or for a feature that does not vary) the plot shows every photo grouped by pile with the spread within a pile
(photographing / detector noise, in % of the mean) against the spread between piles; this is the "how many photos of
one pile" question of task 2. The curve figure has three panels (r, slope, width of the 95 % range of r), every size
labelled with its photos and piles.

**Pile size and the scale.** The relative error of the balance is its rounding error divided by the pile weight (a
balance in 0.5 g steps reads a 20 g pile within about 0.25 g, 1.25 %, and a 3 g pile within 8 %). Piles that all have
the same kernel count differ too little to correlate with anything: vary the pile size for this analysis.

```python
from detkit.weight import SampleTable, feature_correlations, bootstrap_ci, learning_curve
t = SampleTable.from_project(Project.load("runs/mine")).with_weight()
feature_correlations(t, n_boot=1000)                          # the table
bootstrap_ci(t, "kernel_area_px", stat="slope")              # (lo, hi)
learning_curve(t, "kernel_area_px", sizes=[5, 10, 20, 40])   # one dict per size
```
