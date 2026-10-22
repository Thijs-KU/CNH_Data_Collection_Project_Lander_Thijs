# Cheat sheet: `weight features`

Every command starts with `python -m detkit weight features`; `$W` is the workdir (default `runs/default`).
More explanation: [WEIGHT.md](WEIGHT.md) (section "Feature statistics and plots").

## What it prints

1. Header: photos, piles, photos with a weight > 0.
2. Correlation table per feature (r, r2, slope, with `--boot` also the 95 % interval), sorted by |r|.
3. Value table per feature (min, median, mean, p95, max, std). Always printed, also without weights.
4. With `--curve`: learning-curve table per feature.

## Options

| Option | Effect |
|---|---|
| `--workdir DIR` | project folder (default `runs/default`) |
| `--use reviewed\|predictions` | use reviewed boxes (default) or `predictions.json` |
| `--predictions FILE` | predictions file (implies `--use predictions`) |
| `--reviewed-only` | only photos marked reviewed |
| `--crop C` | only photos of this crop |
| `--boot N` | 95 % bootstrap interval of r by resampling piles (0 = off) |
| `--plot FILE` | one overview figure with every feature |
| `--plot-dir DIR` | one larger figure per feature |
| `--format png\|pdf\|svg` | file type of the figures (default png) |
| `--show` | open the figures in windows (needs a display) |
| `--curve F` | learning curve of feature `F`; `a,b` for several, `all` for every feature. Always plotted |
| `--sizes 5,10,20` | photo counts for `--curve` (default: 8 steps up to all photos) |
| `--repeats N` | random subsets per size for `--curve` (default 200) |

## Where the figures go

| Figure | Location |
|---|---|
| overview | the `--plot` file |
| per feature | `--plot-dir/<feature>.<format>` |
| learning curve | `--plot-dir`, else next to `--plot`, else `$W/reports/features` (`learning_curve_<feature>`, `learning_curve_all` for several) |

## Common commands

```bash
# tables only
python -m detkit weight features --workdir $W

# with confidence interval of r
python -m detkit weight features --workdir $W --boot 1000

# all feature plots for the report
python -m detkit weight features --workdir $W --boot 1000 --plot all.png --plot-dir feature_plots

# learning curve of one feature / a few / all
python -m detkit weight features --workdir $W --curve kernel_area_px
python -m detkit weight features --workdir $W --curve count,kernel_area_px
python -m detkit weight features --workdir $W --curve all --sizes 5,10,20,40 --repeats 300

# vector figures for the report
python -m detkit weight features --workdir $W --plot-dir feature_plots --format pdf

# one crop, predicted boxes
python -m detkit weight features --workdir $W --crop wheat --use predictions
```

## Reading the results

* Photos of one pile are not independent; the bootstrap and the learning curve resample whole piles.
* Fewer than 3 weighed photos: no correlation, only the value table and the plots by pile.
* Learning curve: read the width of the 95 % band, not r itself. When it stops getting narrower, more photos add little.
* Scale resolution (0.5 g steps) limits what small weight differences can show; vary the number of kernels per pile.
* A high r does not decide between count, box area and kernel area (they all correlate): compare the calibration error and an intercept near 0.
