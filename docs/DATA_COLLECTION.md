# Collecting data

A **sample** = one physical pile of grain, photographed, then weighed. You get a working detector; your job is to
**design the data-collection process** so that its output can be turned into a weight you can defend.

Data collection is the core of this course, so this page gives no recipe. Before you take a single photo:

This page is task 1 of [STUDENT_TASKS.md](STUDENT_TASKS.md).

1. **Plan.** Decide what you want to estimate for your crop, over what range, and how many samples you need.
   Write the plan down.
2. **Design the process.** Define, step by step, how a sample is prepared, photographed, weighed and recorded,
   so that someone else could repeat it and get comparable data.
3. **Identify the errors.** List everything between the pile on the table and the final weight that could
   make the estimate wrong or noisy.
4. **Measure them.** For each error source, decide whether you can measure its size, and how. Design your
   collection so those measurements are part of it, not an afterthought. Errors you cannot measure: say so and
   argue how large they might be.
5. **Run a small pilot**, look at the result, revise the plan, then collect.

You are graded on how well this plan is reasoned and carried out, and on how honestly you report the uncertainty.

## Recording
Grain moisture is assumed constant for all samples; you do not need to measure or record it.

The app form fields are defined in `detkit/schema.py`. If your plan needs something recorded that is not there
(who took the photo, camera height, ...), add it in the app: Review tab, *Add an extra property*. It is saved per
project and can be imported with `--meta`. A property named `scale_mm_per_px` also gives the area features in mm².

## Import
```bash
python -m detkit import-photos --workdir $W --images my_photos/ --crop wheat
python -m detkit import-photos --workdir $W --images my_photos/ --meta weights.csv     # per-photo metadata
# new crop
python -m detkit import-photos --workdir $W --images corn_photos/ --crop corn --add-class corn
```
`weights.csv`: column `file` plus any keys from `detkit/schema.py`. Re-importing is safe.

Next: check the detections and fill in the metadata in the app ([APP.md](APP.md)). Only saved photos count.
