# Changes to detkit (log)

Every change made to the code base is recorded here, newest first. Append new entries at the top. The other
docs (`APP.md`, `WEIGHT.md`, ...) describe the original toolkit and are left unchanged.

Baseline: the repository at its last commit. Tested with pytest in a Linux sandbox (Python 3.10, Gradio 6.30, plus a
headless-browser run of the app on Gradio 6.28); not run on the Windows `.venv`, and the real detector was not run.

---

## 2026-10-20 (3): detection defaults: score 0.2, tile 1150

* `detkit/project.py`: `DEFAULT_SCORE = 0.2`, `DEFAULT_TILE = 1150` (px).
* Review tab and Inference tab sliders start at these values (before: score 0.5, tile = the project's tile, in the
  Inference tab the best-F1 score of the last eval and the checkpoint's tile). The Inference tab still shows the
  best-F1 score of the newest eval as a hint under the score slider. Choosing another checkpoint now only changes
  the overlap, no longer the tile.
* `detkit predict`: `--score` defaults to 0.2 and `--tile` to 1150 (before: 0.5, and the checkpoint's tile).
* `propose_rfdetr` / `propose_photos` and the scripted Inference `infer` API use the same defaults.
* Not changed: `detkit eval --score` stays 0.05 (a floor, so the F1 curve can find its best score) and eval's tile
  still comes from the manifest / checkpoint; the project's own `tile` (800, used to cut training tiles).

## 2026-10-20 (2): the same normalization for uploads, `predict` and `eval`; new defaults 220 / contrast 1.0

* Defaults changed everywhere: target brightness **220** (was 200), contrast enhancement **1.0** (was 0). They are
  `TARGET_BRIGHTNESS` / `DEFAULT_CONTRAST` in `detkit/preprocess.py` and the start values of all sliders.
  Photos you already normalized keep the settings stored in `meta/warps.json`; apply again to change them.
* `detkit/preprocess.py`: `prepare_photo(image, source, normalize)` (normalizes a RAW photo, leaves photos from a
  project's `photos` folder alone, so nothing is processed twice), `in_project_photos(path)` and
  `before_after(...)` (original | normalized side by side with the median brightness).
* **Inference tab** (`detkit/app/inference.py`, `inference_ui.py`): checkbox "Normalize brightness / contrast first"
  (on), target and contrast sliders. A gallery "Preprocessing: original vs normalized" fills as soon as photos are
  uploaded or a slider is released, before any detection; the detector, the overlays and the results then use the
  normalized photo. *Save as samples* stores the normalized photo in the project and keeps the raw upload in
  `photos_original/` with its settings in `meta/warps.json`, so the Preprocessing tab can redo or undo it
  (`corners.adopt_normalized`). The scripted `infer` API takes `normalize` (default true).
* `detkit predict --normalize [--target-brightness 220] [--contrast 1.0]` and the same flags on
  `detkit eval --model ... --images <raw folder>`. Per photo the log prints `brightness before -> after`;
  `predictions.json` gets `"normalized": {target, contrast, gamma, before, after}` (null = untouched); `predict`
  writes `<name>_pre.jpg` (original | normalized, next to the overlay unless `--no-overlay`), `eval` writes
  `<out>/normalized/<name>_pre.jpg`. Photos taken from a project's `photos` folder (the eval default) are skipped.
  Order with `--canonicalize`: canonicalize first, then normalize (same order as the project photos).
* Tests: `tests/test_preprocess.py` (defaults, CLI flags, `prepare_photo`), `tests/test_app.py` (Inference: only
  uploads are normalized, saved with record and original), `tests/test_predict_single_class.py` (`predict_paths` and
  eval `_run_model` with a stubbed model). Also driven in a headless browser.
* Limits: a very dark photo (median below roughly 100) cannot reach 220 because gamma is capped at 1/6 (the preview
  and the log show the value actually reached). Contrast 1.0 (full CLAHE) also sharpens the kernel edges, so re-check
  detector counts and box areas on normalized photos before relying on the area constant.

## 2026-10-20: "Corners" tab renamed "Preprocessing", brightness / contrast normalization

* The tab is now called **Preprocessing** (label in `detkit/app/ui.py`). Entries below that say "Corners tab" mean
  this tab; the files keep their names (`detkit/app/corners.py`, `corners_ui.py`).
* `detkit/preprocess.py`: `TARGET_BRIGHTNESS = 200`, `brightness(image)` (median CIELAB lightness, 0-255) and
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
