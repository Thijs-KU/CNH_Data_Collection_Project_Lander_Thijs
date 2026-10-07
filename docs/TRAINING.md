# Training the detector

`detkit train` fine-tunes RF-DETR (`nano < small < base < medium < large`) on photos you reviewed in the app.

> **Optional, not part of the course.** The course is about designing the data collection so that the provided
> baseline model can be used to estimate weight. You do not need to retrain it; this page is for those who want to.

## Retrain
```bash
source .venv/bin/activate      # once in every new terminal (see README)
W=runs/mine
python -m detkit split --workdir $W --from-reviewed --val-sources photoA.jpg,photoB.jpg   # or --val-frac 0.2
python -m detkit train --workdir $W --name run1 --epochs 60 --lr 1e-4
python -m detkit eval  --workdir $W --gt val --model $W/checkpoints/run1/final --score 0.05
python -m detkit app   --workdir $W --ckpt $W/checkpoints/run1/final                       # better proposals next time
```
`final/` is the best-eval checkpoint. You need at least 2 reviewed photos, and every class in both train and val.

## Flows
| | cpu | gpu | hf |
|---|---|---|---|
| `init --flow` | `cpu` | `gpu` | `hf` |
| install | `uv sync --extra cpu --extra app` | `uv sync --extra gpu --extra app` | either |
| default model / batch / workers | `nano` / 2 / 0 | `small` / 4 / 2 | `small` / 4 / 2 |
| use for | small fine-tunes (slow) | everything | sharing a model |

Defaults are stored in `<workdir>/project.json`; override any of them:
```bash
python -m detkit train --workdir $W --model small --batch-size 4 --num-workers 2 --epochs 60 --lr 1e-4 --device cuda
```
Other flags: `--name`, `--grad-accum`, `--no-augment`, `--no-load-best`, `--device cpu`. `--tile` is set at `init`.

### Hugging Face
* Base weights download automatically on first `train`/`app` run (internet, no token).
* Share a model (token with write access):
  `python -m detkit train --workdir $W --name run1 --push-to-hub --hub-model-id <user>/grain-detector`
* No GPU: run the same `train` command on Colab / a classmate's machine and copy back `checkpoints/run1/final`.

## Optional experiment
Compare `--no-augment` against the default on the same val photos and report the difference.

## Troubleshooting
| symptom | fix |
|---|---|
| `coco/train/labels.json missing` | run `split --from-reviewed` |
| `class 'x' has 0 boxes in val` | choose `--val-sources` so each crop has a val photo |
| `CUDA out of memory` | smaller `--batch-size` or `--model` |
| very slow on laptop | `--model nano --batch-size 2 --num-workers 0`, fewer tiles |
| huge CUDA torch pulled on a laptop | use `--extra cpu` (not `gpu`) |
| Windows: training hangs or a dataloader worker crashes | `--num-workers 0` |
| `TrainingArguments has no warmup_ratio` | keep the pinned `transformers==5.12.1` |

The shipped baseline's exact training recipe, expected curve and scores: `weights/grain_rfdetr_small_v1/TRAINING.md`.

Baseline model: https://huggingface.co/niru-5/manama_course_rfdetr_grain_detection. Dataset: https://huggingface.co/datasets/niru-5/manama_course_dataset_grain_detection.
