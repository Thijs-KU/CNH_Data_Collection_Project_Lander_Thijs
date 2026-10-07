# vendor/

Verbatim copies from `../../vision-intern` (commit `e328fde`, MIT), kept only so this project
runs without the parent repo. Edits are listed below. Do not edit
casually: re-sync from vision-intern instead.

## train_rfdetr.py edits (public-dataset run)
- `train()` gained `eval_batch_size` and `warmup_ratio` (was hard-coded 0.05) and now passes `seed`/`data_seed` to
  `TrainingArguments` (the seed argument was previously unused for training). Defaults keep the old behaviour
  except that the Trainer seed is now 1337 instead of the transformers default 42.
