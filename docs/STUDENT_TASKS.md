# Student tasks

Each group (2 people) gets **one crop** and follows it from physical samples to a weight you can defend. The tasks
build on each other: what you measure in task 1 is what you use in tasks 2 and 3, the procedure you write is tested
in task 4, and the model you end up with is used in task 6.

All tasks use the provided detector; none of them requires retraining it. How to solve each task is up to you:
justify your choices.

| # | Task | Code |
|---|---|---|
| 1 | Plan the data collection: errors, how to prevent them, how to measure them | - |
| 2 | Which feature tells you the weight, and how many samples you need | `detkit/weight/stats.py`: `learning_curve`, `bootstrap_ci` |
| 3 | The bounds of your predictions, and how the error propagates | `detkit/weight/models.py`: `low_g`/`high_g`; `stats.py`: `propagate` |
| 4 | Your team-mate repeats the procedure on a small sample: do the conclusions still hold? | - |
| 5 | Report | - |
| 6 | Deployment: estimate the weight of a sample we give you | - |

Code tasks: `git grep -n "TODO(student)" -- detkit`. Self-check: `python -m pytest tests/test_student_tasks.py -q -rs`
(skipped = not done yet; a passing test only checks the interface, not whether your approach is sound).

## 1. Plan the data collection
A **sample** is one physical pile of grain: photographed, then weighed. Before you collect, think it through
([DATA_COLLECTION.md](DATA_COLLECTION.md)):
* What kinds of errors can happen while collecting data (the photo, the pile, the balance, the recording)?
* How would you prevent each of them?
* How would you set up experiments to **measure** those errors? (For example: photograph the same pile several
  times, weigh it again, count a pile by hand.)
* How can you be sure your procedure works? Run a small pilot, look at the result, revise.

Discuss the procedure together and write it down step by step. Then **one** of you collects the data for tasks 2
and 3; the other one repeats it in task 4, so the procedure must be complete enough to follow from the text alone.
Record what your plan needs: the app form takes extra properties (Review tab, *Add an extra property*), e.g. a
`collector` property, so you can tell the two persons' data apart later.
Name a property `scale_mm_per_px` and the area features are also given in mm² (`total_area_mm2`, `mean_area_mm2`).

## 2. Which feature correlates best with weight, and how many samples do you need?
You have collected samples and have the detector's boxes for them. The features per photo are `count`,
`total_area_px`, `mean_area_px`, `median_area_px`, `mean_diag_px` (and the mm² ones with a scale); see
`detkit/weight/features.py`.
* Show how the different features correlate with the measured weight, and pick one.
* How sure are you about that correlation? Implement `bootstrap_ci` (a confidence interval of `r` or the slope).
* How many samples do you need to be sure about it? Implement `learning_curve`.
* How many photos of the **same** sample do you need? Compare the spread between photos of one pile with the
  spread between piles.
* Photos of the same pile (same `pile_id`) are not independent samples: keep that in mind when you resample.

Then calibrate the model that uses your feature: `detkit weight calibrate --model count_x_constant` (weight = count
x grams per kernel) or `--model area_x_constant` (weight = box area x grams per px²). See [WEIGHT.md](WEIGHT.md).

## 3. Bounds and error propagation
* What could be the bounds of your predictions? Make the models return `low_g` / `high_g` (`TODO(student)` in
  `detkit/weight/models.py`); the app's Inference tab shows them. `evaluate_model` (`detkit/weight/stats.py`) gives
  the per-photo errors (predicted vs balance) to start from.
* How does the error propagate across the different stages (camera / scale, detector, kernel weight, balance, ...)?
  Implement `propagate`, fed with the error of each stage **as you measured it in task 1**. Which stage dominates,
  and what would you improve first?
* Compare the two: the bounds from the model's spread (top-down) and the total from your stages (bottom-up). Do
  they agree? If not, what is missing? (Careful not to count the same error twice.)

## 4. Would the conclusions hold if your team-mate repeats it?
The person who did **not** collect the data in tasks 2-3 now collects a small new sample, following only the written
procedure. Apply the model from tasks 2-3 to it, without re-fitting (`evaluate_model(model, new_samples)` gives the
errors per photo, the bias and MAPE; `detkit weight predict` prints the same):
* Do the new weights fall inside your bounds from task 3?
* Calibrate the constant on the new sample alone: is it the same within its uncertainty?
* Does the chosen feature still correlate best? Do your task 2 conclusions still hold?
* If something differs: is it the person (the way they photograph, pile or weigh), or chance with a small sample?
  How would you tell? Where was the procedure unclear? Improve it.

## 5. Report
One report with all your learnings and findings: the plan and the procedure, the errors and how big they are,
the chosen feature and the evidence, the bounds and the error budget, the repeat by your team-mate, and what you
would change.

## 6. Deployment
Run the app on your laptop with your final model and constants, reachable over the network
(`detkit app --host 0.0.0.0`, see [APP.md](APP.md#use-from-a-phone)). We bring a grain sample of your crop,
connect to your network, take photos with our phone following your procedure, and upload them in the
**Inference** tab: it predicts the boxes and applies your model.
* The result is the weight **with its bounds**, as the Inference tab shows it. No re-fitting on the sample.
* Your procedure must be clear enough for us to take the photos the way your model expects them.

This part is to have a small demo at the end of each report presentation. The end value can go wrong, but don't be worried about it. It is important to focus on the process behind it. The grade itself depends on how you have tackled each section and what you have learnt from it. 