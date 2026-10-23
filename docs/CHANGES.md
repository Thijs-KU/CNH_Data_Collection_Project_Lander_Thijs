# Changes to detkit (log)

This file lists how the code base differs from the original project, not the history of how it got there: when a
setting or behaviour changed more than once, only the end result is described. The other
docs (`APP.md`, `WEIGHT.md`, ...) describe the original toolkit and are left unchanged.

Baseline: the repository at its last commit. Tested with pytest in a Linux sandbox (Python 3.10, Gradio 6.30, plus a
headless-browser run of the app on Gradio 6.28); not run on the Windows `.venv`, and the real detector was not run.

---

## Defaults: detection score 0.2, tile 1000, brightness 230, contrast 1.0

* `detkit/project.py`: `DEFAULT_SCORE = 0.2`, `DEFAULT_TILE = 1000` (px).
* Review tab and Inference tab sliders start at these values (originally: score 0.5, tile = the project's tile, in
  the Inference tab the best-F1 score of the last eval and the checkpoint's tile). The Inference tab still shows the
  best-F1 score of the newest eval as a hint under the score slider. Choosing another checkpoint now only changes
  the overlap, no longer the tile.
* `detkit predict`: `--score` defaults to 0.2 and `--tile` to 1000 (originally 0.5, and the checkpoint's tile).
* `propose_rfdetr` / `propose_photos` and the scripted Inference `infer` API use the same defaults.
* `detkit/preprocess.py`: `TARGET_BRIGHTNESS = 230` and `DEFAULT_CONTRAST = 1.0` are the start values of all
  brightness / contrast sliders and the defaults of `--target-brightness` / `--contrast` on `predict` and `eval`.
  Photos you already normalized keep the settings stored in `meta/warps.json`; apply again to change them.
* Not changed: `detkit eval --score` stays 0.05 (a floor, so the F1 curve can find its best score) and eval's tile
  still comes from the manifest / checkpoint; the project's own `tile` (800, used to cut training tiles).

---

## Inference tab: kernel area column and green kernel pixels

* The results table (and CSV) get a column `kernel_area_px` next to `total_area_px` (box area): the kernel pixels
  inside the boxes, the feature the `kernel_area_x_constant` model uses (`detkit/app/inference.py`).
* New checkbox "Show the kernel pixels (green) inside the detected boxes" (on by default): the detection images tint the
  pixels that `weight/features.py: kernel_mask` counts green (`inference.overlay(kernel_pixels=...)`,
  `run_inference(kernel_overlay=...)`; off by default in the scripted API).
* `detkit weight calibrate` prints a WARNING when photos of one pile (same `pile_id`) have different
  `total_weight_g` (`SampleTable.inconsistent_piles`): one pile has one balance reading, so it is probably a typo, and a
  single wrong weight inflates the spread and with it `low_g` / `high_g`.

---

## `detkit eval`: scatter of the kernel area (automatic)

* `detkit/cmd_eval.py`, no flag needed: per photo `eval` compares the kernel pixels (`weight/features.py: kernel_mask`, the
  `kernel_area_px` feature) of the predicted boxes (score >= `--op-score`, all classes pooled) with those of the GT
  boxes, like `area_scatter.png` does for the box area. It reads the photos (`--images`, default the project's
  photos); when none are found, or with `--canonicalize` (its frame differs from the boxes'), it is skipped with a
  note. `--no-kernel-area` turns it off (`--kernel-area` is still accepted and does nothing).
* Written to the report folder: `kernel_area.csv`, `kernel_area_scatter.png`, a `kernel_area` block in `eval.json`
  and a section "Kernel-pixel area error" in `eval.md`. It needs the photos and adds some time per photo.
* `detkit/evaluate.py`: `kernel_area_table`, `write_kernel_area`. Test: `test_kernel_area_scatter_in_report`.

---

## Inference tab: warp new photos to the A4 sheet

* New accordion "Warp new photos to the A4 sheet (optional)" in the Inference tab (`detkit/app/inference_ui.py`): pick an
  uploaded photo, click the 4 corners of the sheet (same clicking as the Preprocessing tab, `detkit/app/corners.py`),
  check the warped preview, *Use this warp* (or *Remove the warp of this photo*). Several photos can have a warp;
  uploads without one are used as before. The preview gallery shows the warp (and the brightness step if it is on).
* `inference.prepare_photos` / `run_inference` take `corners` ({uploaded file name: 4 points in pixels of the upload}):
  warp to the 900x1200 canvas first, then the brightness step. Photos already in the project's photos folder are never
  warped again. The scripted `infer` API takes `corners_json`.
* *Save as samples* stores the warped photo (its size is the canvas size) and keeps the raw upload in
  `photos_original/` with the warp (and brightness) in `meta/warps.json`, so the Preprocessing tab can redo or undo it
  (`corners.adopt_preprocessed`; `adopt_normalized` is its brightness-only case). `scale_mm_per_px` is set when the
  project has that property, as for a warp in the Preprocessing tab.
* The ROI field is now in pixels of the photo after the warp. Tests: `test_inference_warps_uploads_with_clicked_corners`
  (logic only; the Gradio layout was not run in the sandbox).

---

## Weight models: spread and bounds `low_g` / `high_g` (task 3, part 1)

* `detkit/weight/models.py`, `FeatureTimesConstant.fit`: besides the constants it stores `spreads` per crop (and pooled
  for `default`): `sd` of the per-pile mean ratios weight / feature, `n` = number of piles, `photos`. Photos of one
  pile (same `pile_id`, one shared weight) are not independent, so the unit is the pile; a photo without `pile_id` is
  its own pile. A crop with fewer than 2 piles gets no spread. `params()` / `set_params()` carry the spreads (files
  without them still load).
* `predict` returns a 95 % prediction interval for a new pile: `weight +- t(n-1) * sqrt(1 + 1/n) * sd * feature`,
  `low_g` not below 0. Student-t factors come from a small table (`t_factor`), so no scipy is needed. Constant and
  spread come from the same crop (`default` for an unknown crop); with no spread the bounds stay empty and the note
  says so.
* Tests: `tests/test_weight.py` (`test_fit_stores_spread_per_pile`, `test_predict_gives_95_percent_prediction_interval`);
  `test_task3_models_give_bounds` in `tests/test_student_tasks.py` now runs instead of skipping.
* `detkit weight calibrate` also saves the spreads, in `weight_config.json` under `spreads` (`{<constants key>:
  {<crop>: {sd, n, photos}}}`; the CLI prints them), and the CLI and the Inference tab rebuild the model with them,
  so the tab shows `low_g` / `high_g`. Editing a constant by hand in the tab's constants table removes the spread of
  that crop when the value changes by more than 0.1 % (no bounds rather than misleading ones). A config written
  before this change has no spreads: calibrate again.
* Not done: `propagate` (task 3, part 2).

---

## 2026-10-21 (3): `weight features`: value table

**Why.** With weights the command only printed r and the slope; the distribution of each feature (the table that
showed min / mean / std / max) was only printed when there was no correlation.

* A table of every feature over all photos is now always printed: min, median, mean, p95, max, std.
* `weight features` now saves the figures (overview + one per feature) in `<workdir>/reports/features` when no plot option
  is given, together with the learning curve of every feature; `--no-plots` gives tables only.
  Relative `--plot` / `--plot-dir` / `--out` paths are now placed inside `<workdir>/reports`, so figures never end up
  in the folder the command was started from.
* New `python -m detkit weight kernels`: example figures of the kernel-pixel feature (photo with boxes and kernel pixels
  plus close-ups of single kernels), for the report. Test: `test_kernels_cli`.
* New `docs/CHEATSHEET_FEATURES.md`: all options, output locations and example commands of `weight features`.
* `--curve` now always draws the learning curve (before, only together with `--plot` / `--plot-dir`): saved in
  `--plot-dir`, next to `--plot`, or else in `<workdir>/reports/features`. `--curve a,b` or `--curve all` handles
  several features and also writes one comparison figure (`learning_curve_all`).
* Tests: `tests/test_weight_stats.py` (table with and without weights).

---

## 2026-10-21 (2): task 2 statistics: feature correlations, `bootstrap_ci`, `learning_curve`

**Why.** Task 2 asks which feature follows the measured weight best and how sure and how many samples that needs.
The comparison of the features had no code, and `bootstrap_ci` / `learning_curve` were stubs.

* `detkit/weight/stats.py`:
  * `feature_correlations(samples, crop, features, n_boot)`: per numeric feature r, r², least-squares slope and
    intercept (+ bootstrap interval of r), best first. Features that do not vary are left out.
  * `bootstrap_ci(samples, x_key, ..., stat="r"|"slope", n_boot, alpha, seed)`: percentile bootstrap that resamples
    whole piles (`pile_id`; a photo without one is its own pile), because photos of one pile are not independent;
    resampling photos would give too narrow an interval. `None` with fewer than 3 weighed photos or 2 piles.
  * `learning_curve(samples, x_key, sizes, ..., repeats, seed)`: per size, random subsets of whole piles; mean, std
    and central 95 % range of r and of the slope over the repeats. At the full size the spread is 0 by construction.
  * A feature that is constant up to rounding error gives r = NaN (not a warning from `polyfit`).
* `python -m detkit weight features --workdir W [--crop C] [--boot N] [--plot PNG] [--plot-dir DIR] [--show]`: the table above
  and plots. `--plot` = one overview PNG, `--plot-dir` = one larger PNG per feature, `--show` = open the figures.
  Points are coloured by pile; with weights each plot shows the weight against the feature, the free least-squares
  line and the line through the origin (the one `weight = feature x constant` uses, its slope is the constant).
  The plots do not need weights or a correlation: with fewer than 3 photos with a weight > 0, or for a feature that
  does not vary, the plot shows the value of the feature for every photo (mean marked) and the console prints
  mean / std / min / max per feature. Weights <= 0 are left out (and reported).
* `--curve FEATURE [--sizes 5,10,20] [--repeats N]`: prints and plots (with `--plot` / `--plot-dir`) the learning
  curve of that feature: r and slope against the number of photos, mean and 95 % range over random subsets of whole
  piles. Read the width of the band, not the value of r: it has settled when the band stops shrinking.
* Plots for the report (`detkit/weight/plots.py`, new): **every pile has its own colour and marker** (photos of one
  pile share both; a legend lists the piles and their photo count; a photo without `pile_id` is a pile of its own).
  Photos of one pile have the same weight, so they sit in one horizontal row; the hollow diamond is the pile mean.
  Titles carry r, its interval, R² and the numbers of photos and piles; axes carry units (kernels, px², mm²).
  Without weights the plot groups the photos by pile and prints the spread within a pile (photographing / detector
  noise, % of the mean) against the spread between piles. The learning curve has three panels (r, slope, width of
  the 95 % range of r), every size labelled with its photos and piles, the full-size point drawn as a reference
  instead of a band (it has no spread by construction). `--format png|pdf|svg` for the per-feature and curve plots.
* `numeric_features(samples)` (stats.py): the feature keys every photo has as a number.
* Tests: `tests/test_weight_stats.py` (ranking, pile-wise vs photo-wise intervals, reproducibility, curve settles,
  edge cases, CLI); the three task 2 checks in `tests/test_student_tasks.py` now run and pass instead of skipping.
* Not done: `propagate` (task 3).

---

## 2026-10-21: kernel-pixel area feature (`kernel_area_x_constant`)

**Why.** The box features (width, height, box area) depend on how the kernels lie. A box can only be drawn along the
image axes, so a kernel lying diagonally gets a much larger box than the same kernel lying straight (at 45 degrees
roughly twice the area), while its real size is the same. A pile with more diagonal kernels therefore gets a bigger
box area without containing more popcorn, which adds noise to the weight estimate. Counting only the pixels that really
are kernel does not depend on the orientation: a kernel covers the same number of pixels whichever way it points.

* `detkit/weight/features.py`: `kernel_mask(image, bboxes)` finds the pixels inside the boxes that are kernel (colour
  distance to the paper, Otsu inside/around the boxes, per-box close + largest blob + convex hull; overlapping boxes
  count once). `image_features(detections, meta, image=None)` adds `kernel_area_px`, `mean_kernel_area_px`,
  `median_kernel_area_px`, `kernel_fill_ratio` (+ `kernel_area_mm2`, `mean_kernel_area_mm2` with a scale) when given
  the preprocessed photo; without `image` nothing changes. `SampleTable.add(..., image=)`; `from_project` reads
  `<workdir>/photos/<file>` (`kernel_area=False` skips).
* `detkit/weight/models.py`: `KernelAreaTimesConstant` (`kernel_area_x_constant`, constants in
  `kernel_area_constants` of `weight_config.json`); `detkit weight calibrate|predict` accept it.
* Inference tab: `estimate_row(..., image)` gets the prepared photo, so the model can be chosen there. The constants
  table in the tab still has only the count and box-area columns: calibrate this one with `detkit weight calibrate`.
* On the 4 sample photos the kernel pixels are about 70 % of the box area (fill ratio 0.69 to 0.73).
* Tests: `tests/test_weight.py` (synthetic discs, overlapping boxes, empty cases, table + model).

---

## 2026-10-20 (2): the same normalization for uploads, `predict` and `eval`

* Default target brightness and contrast: see the defaults entry at the top.
* `detkit/preprocess.py`: `prepare_photo(image, source, normalize)` (normalizes a RAW photo, leaves photos from a
  project's `photos` folder alone, so nothing is processed twice), `in_project_photos(path)` and
  `before_after(...)` (original | normalized side by side with the median brightness).
* **Inference tab** (`detkit/app/inference.py`, `inference_ui.py`): checkbox "Normalize brightness / contrast first"
  (on), target and contrast sliders. A gallery "Preprocessing: original vs normalized" fills as soon as photos are
  uploaded or a slider is released, before any detection; the detector, the overlays and the results then use the
  normalized photo. *Save as samples* stores the normalized photo in the project and keeps the raw upload in
  `photos_original/` with its settings in `meta/warps.json`, so the Preprocessing tab can redo or undo it
  (`corners.adopt_normalized`). The scripted `infer` API takes `normalize` (default true).
* `detkit predict --normalize [--target-brightness 230] [--contrast 1.0]` and the same flags on
  `detkit eval --model ... --images <raw folder>`. Per photo the log prints `brightness before -> after`;
  `predictions.json` gets `"normalized": {target, contrast, gamma, before, after}` (null = untouched); `predict`
  writes `<name>_pre.jpg` (original | normalized, next to the overlay unless `--no-overlay`), `eval` writes
  `<out>/normalized/<name>_pre.jpg`. Photos taken from a project's `photos` folder (the eval default) are skipped.
  Order with `--canonicalize`: canonicalize first, then normalize (same order as the project photos).
* Tests: `tests/test_preprocess.py` (defaults, CLI flags, `prepare_photo`), `tests/test_app.py` (Inference: only
  uploads are normalized, saved with record and original), `tests/test_predict_single_class.py` (`predict_paths` and
  eval `_run_model` with a stubbed model). Also driven in a headless browser.
* Limits: a very dark photo (median below roughly 100) cannot reach a high target because gamma is capped at 1/6 (the preview
  and the log show the value actually reached). Contrast 1.0 (full CLAHE) also sharpens the kernel edges, so re-check
  detector counts and box areas on normalized photos before relying on the area constant.

## 2026-10-20: "Corners" tab renamed "Preprocessing", brightness / contrast normalization

* The tab is now called **Preprocessing** (label in `detkit/app/ui.py`). Entries below that say "Corners tab" mean
  this tab; the files keep their names (`detkit/app/corners.py`, `corners_ui.py`).
* `detkit/preprocess.py`: `TARGET_BRIGHTNESS` (default 230, see the defaults entry), `brightness(image)` (median CIELAB lightness, 0-255) and
  `normalize_brightness(image, target, contrast)`.
  * One gamma curve on the lightness channel puts the median at `target`; gamma keeps black and white fixed, so
    nothing clips, and is limited to [1/6, 6]. Colours (a, b) are not touched. Because converting back to RGB clips
    very saturated colours, the real output is measured and gamma corrected up to 3 times; `after` in the returned
    info is that measured value. White paper reaches the target; a saturated colour photo may stay a bit below a
    high target.
  * `contrast` 0-1 mixes in a CLAHE of the lightness channel (clip 2.0, 8x8 tiles) before the brightness step.
  * The size never changes, so saved boxes stay valid.
* `detkit/app/corners.py`: the pipeline of a stored photo is always *original -> recorded warp -> recorded
  brightness*, rebuilt from `photos_original` and `meta/warps.json`, so changing a setting never stacks.
  * `normalize_photo`, `reset_photo_brightness`, `normalize_all`, `normalized_preview` (stores nothing).
  * A brightness change keeps boxes and metadata (unlike the warp). A redone warp re-applies the recorded
    brightness. A photo that was never warped is normalized too: its raw copy goes to `photos_original`, and its
    record has no `corners` (only warped photos show `warped` in the list).
  * New list columns `normalized` (yes / -) and `brightness` (current median lightness), appended at the end.
* `detkit/app/corners_ui.py`: "Brightness and contrast" panel with a target slider (60-240), a contrast slider
  (0-1), a live preview (updates on slider release and when a photo is opened), *Apply to this photo*, *Apply to all
  photos* and *Remove brightness change from this photo*.
* `meta/warps.json` records get a `normalize` entry `{target, contrast, gamma, before, after}`; records of photos that
  were only normalized have `source` and `normalize` but no `corners`.
* Re-importing a raw source file after normalizing is recognised (same original), like after a warp.
* Tests: `tests/test_corners.py` (target reached, contrast widens the spread, boxes kept, no stacking, reset, rewarp
  keeps brightness, apply to all, re-import). Also driven in a headless browser (Gradio 6.28): preview, apply,
  apply to all and reset all work, no console errors.
* Limits:
  * Brightness is a single global curve: uneven lighting or shadows on the sheet are not removed, and the median
    includes whatever is not paper in a photo that was not warped, so warp first.
  * Only photos already in the project are normalized. New Inference tab uploads, `detkit predict` and
    `detkit eval --images <folder>` are not, so they should be normalized the same way before they are compared
    with the project photos (not wired in yet; `Preprocessor.apply` is the intended place).
  * Already reviewed boxes stay valid, but the detector's proposals may change on a normalized photo.

## 2026-10-19 (4): one class per photo

A photo shows one grain type, but the detector often gives one kernel two or three boxes with different classes.
Now only the most prominent class of each photo is kept.

* `detkit/merge.py`: new `keep_dominant_class(dets)`. Prominence = sum of the box scores per class (ties: more
  boxes, then smaller label). Boxes of every other class are erased, so a kernel with a popcorn and a pumpkin box
  keeps the popcorn one. A kernel that only has a box of another class is dropped with it.
* `detkit/predict.py`: `predict_image(..., single_class=True)` applies it after the per-class overlap suppression.
  This covers the Review tab proposals (single and batch), the Inference tab, `detkit predict` and
  `detkit eval --model`. `predict_paths(..., single_class=True)` passes it through.
* `detkit/cli.py`, `detkit/cmd_eval.py`: `--keep-all-classes` on `detkit predict` and `detkit eval` returns every
  class again (use it to measure the raw detector). The app always filters.
* Effect on eval: the detector metrics now describe the filtered pipeline. If a photo's dominant class comes out
  wrong, all its boxes count against that photo.
* Tests: `tests/test_merge.py` (3 new), `tests/test_predict_single_class.py` (`predict_image` with a stubbed model).
* Open choice: relabel boxes of the other classes to the dominant class instead of erasing them (keeps a kernel
  that was only seen as another class, but also keeps stray boxes).

## 2026-10-19 (3): manual A4 warp and the Corners tab

Flattens the A4 sheet in a photo to a fixed 900x1200 canvas, so pixel areas do not depend on the camera distance.

* `detkit/preprocess.py`: `CANVAS_SIZE = (900, 1200)`, `SHEET_MM`, `MM_PER_PX`, `SCALE_MM_PER_PX` (0.2403),
  `WARPS_FILE`, `order_corners(pts)` (4 clicks in any order -> top-left, top-right, bottom-right, bottom-left of the
  portrait sheet; rejects non-convex input) and `warp_to_canvas(image, corners)` (perspective warp at 2x, shrunk
  with INTER_AREA; returns the image and corners / matrix / canvas / mm_per_px).
* `detkit/app/corners.py` (logic, no gradio) and `detkit/app/corners_ui.py` (layout): new **Corners** tab.
  * The list shows the same photos as the Review tab with the columns file, warped (`warped` / `raw`), status,
    boxes and size. It refreshes when the tab is opened and with *Refresh list*.
  * Click a row, click the 4 sheet corners (clicks are mapped back to the pixels of the stored photo, no EXIF
    rotation), check the warped preview, press **Warp this photo**.
  * The photo is replaced in place: same file name, so crop / pile_id / weight / manual_count stay attached.
  * The raw photo is copied to `<workdir>/photos_original/` (first warp only); opening a warped photo shows that
    original, so a warp can be redone from it.
  * Boxes saved for the photo are in the old frame: they are cleared and the photo becomes unreviewed. They are kept
    under `discarded` in `<workdir>/meta/warps.json`, which also stores the corners, matrix, canvas and mm/px.
  * The stored image size is updated to 900x1200. If the project has an extra property named `scale_mm_per_px`, it
    is set to 0.2403 (geometric mean of 0.2333 and 0.2475 mm/px), so `total_area_mm2` is right.
* `detkit/app/ui.py`: adds the tab (`build_corners_tab(project, wd, tab)`).
* `detkit/cmd_data.py`: `import-photos` recognises a photo warped in place (same source as its saved original), so
  re-importing the same raw file is skipped instead of refused.
* Tests: `tests/test_corners.py` (corner ordering, warp round trip, click mapping, list, in-place warp, redo, scale,
  re-import, tab builds). The tab was also driven in a headless browser: clicks land on the right pixels, also for a
  photo with an EXIF rotation tag.
* Limits: 900x1200 is 3:4 and A4 is 1:1.414, so the picture is stretched about 6% in one direction (areas still
  convert with one constant). Kernels are about 40 px wide in this frame: check the detector on warped photos.
  `import-photos` and the Review upload still copy files raw: warp them afterwards in the Corners tab and do not mix
  raw and warped photos in one project. The Inference tab does not warp uploads yet (needed for the task 6 demo).
  The 4 already reviewed photos in `runs/mine` are raw (old frame).

## 2026-10-19 (2): `--canonicalize` and a frame check for `predict` / `eval`

* `detkit/predict.py`, `detkit/cli.py`: `detkit predict --canonicalize` runs the preprocessing on every photo first
  (only for RAW photos, never for photos already in `<workdir>/photos`). `predictions.json` entries get
  `"canonicalized": true/false` (whether that run did it).
* `detkit/cmd_eval.py`: `detkit eval --model ... --canonicalize` does the same. New check `_check_frame`: each image
  must have the size its ground-truth boxes were made on, otherwise eval stops with a message (exit code 1) instead
  of producing meaningless metrics. It also protects `--gt path/to/labels.json`.
* Tests: `tests/test_preprocess.py` (frame check, CLI flags).

## 2026-10-19 (1): preprocessing hook, re-import check, OpenCV

* `detkit/preprocess.py` (new, imports OpenCV): class `Preprocessor` and the shared instance `PREPROCESSOR`.
  * `canonicalize(image)`: geometric preprocessing (may change the size), applied once when a photo enters the
    project. For now it returns the image unchanged; set `PREPROCESSOR.canonicalizer` to switch it on everywhere.
  * `ingest(src, dst)`: puts a photo in the project in canonical form; a byte-exact copy while nothing changes it.
  * `canonical_path(src, out_dir)`: the same for read-only use (Inference uploads); returns `src` while unchanged.
  * `apply(image)`: photometric steps (BGR, same size, raises if a step changes the size). Not wired into
    `predict_image` yet.
* Wiring (no behaviour change while the canonicalizer is off):
  * `detkit/cmd_data.py`: `import-photos` stores photos with `PREPROCESSOR.ingest`.
  * `detkit/app/logic.py`: `import_photos(paths, photos_dir, canonicalize=True)` (used by the Review upload);
    `canonicalize=False` copies as is.
  * `detkit/app/inference.py`: `run_inference` uses `PREPROCESSOR.canonical_path`, so detection, overlay and results
    share the frame of the stored photos; `save_samples` passes `canonicalize=False` (no double warp).
* Re-import check in `detkit/cmd_data.py`: `import-photos` no longer compares the source byte for byte with the stored
  photo (never equal once photos are changed). Each ingest records the sha256 of its source and whether it was
  canonicalized in `<workdir>/meta/photo_sources.json`. Photos without a record are compared byte for byte, which only
  holds while preprocessing is off; with it on they count as different and need `--force`. Changing the canonicalizer
  itself is not detected: re-import with `--force`.
* `pyproject.toml`: new dependency `opencv-python-headless`. Run `uv sync` (it also updates `uv.lock`) or
  `pip install opencv-python-headless`; until then the code that imports `detkit.preprocess` fails with a
  missing-module error.
* Tests: `tests/test_preprocess.py` (new), `tests/test_data.py` (re-import with a canonicalizer).

## Files

New: `detkit/preprocess.py`, `detkit/app/corners.py`, `detkit/app/corners_ui.py`, `docs/CHANGES.md`,
`tests/test_preprocess.py`, `tests/test_corners.py`, `tests/test_predict_single_class.py`.

Changed: `detkit/app/inference.py`, `detkit/app/logic.py`, `detkit/app/ui.py`, `detkit/cli.py`, `detkit/cmd_data.py`,
`detkit/cmd_eval.py`, `detkit/merge.py`, `detkit/predict.py`, `pyproject.toml`, `tests/test_data.py`,
`tests/test_merge.py`.

## Data written by the new features (inside the workdir)

| Path | Written by | Content |
|---|---|---|
| `meta/photo_sources.json` | `import-photos` | sha256 of each source file + whether it was canonicalized |
| `meta/warps.json` | Preprocessing tab | per changed photo: original file name, warp (corners, matrix, canvas, mm/px, discarded boxes), brightness settings |
| `photos_original/` | Preprocessing tab | the raw photo of every warped or normalized photo |
