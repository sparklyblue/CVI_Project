# Pairwise Crop CNN Guide

This folder contains the experimental pairwise CNN approach for classifying one
animal detection as `static` or `moving`.

Unlike the feature-based random-forest baseline, this model learns directly
from thermal image crops. It compares the current animal observation with a
nearby observation from the same recovered track.

## Project Status

This approach was implemented and tested as an alternative to the motion
baseline. Training and automatic tuning were substantially slower than the
feature-based approach, and the tested candidates were not promising enough to
replace the random forest.

No final pairwise CNN weight is therefore presented as a validated project
result. A local file at `dist/pairwise_crop_cnn_smoke/model.pt` may exist after
running the smoke test, but it is only a functionality check and is
intentionally ignored by Git. The validated weight committed for the movement
task is:

```txt
dist/motion_baseline/model.joblib
```

See `motion_baseline/README.md` for the selected model and its results.

## Setup

All commands are expected to be run from the repository root.

Create and activate a virtual environment, then install the shared project
requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

PyTorch is required. The script automatically uses CUDA when a compatible GPU
and CUDA-enabled PyTorch installation are available. Use `--device cpu` or
`--device cuda` to select a device explicitly.

## Required Data

The thermal images are not committed because of their size. Follow the download
instructions in the repository root `README.md` and prepare the data once:

```bash
python3 build_labels.py
python3 filter_label.py
python3 rebalance_splits.py
```

The pairwise CNN then uses:

```txt
images_thermal/images/{train,val,test}/
labels_filtered/{train,val,test}/
mots/*_gt.txt
```

The filtered labels provide the static/moving target. MOTS annotations are used
to recover track IDs. Species is only used while matching boxes back to tracks;
it is not a CNN input feature.

## How The Model Works

For each animal detection:

1. The filtered box is matched back to its MOTS track.
2. A nearby previous or future detection from the same track is selected.
3. One shared local crop region is read from both thermal images.
4. Three CNN input channels are created: current crop, neighbor crop, and their
   absolute difference.
5. Numeric context is added, including frame gap, temporal direction, raw box
   displacement, box area, edge distance, and track-match IoU.
6. The CNN outputs a probability that the current animal is moving.
7. When several pairs belong to one target detection, their probabilities are
   averaged into one per-animal prediction.

If no temporal neighbor is available, the current crop is duplicated and the
metadata records that no neighbor exists.

## Main Files

| File | Purpose |
| --- | --- |
| `train_pairwise_crop_cnn.py` | Small root entrypoint that calls the package CLI. |
| `pairwise_crop_cnn/cli.py` | Full training, tuning, evaluation, and export pipeline. |
| `pairwise_crop_cnn/pairs.py` | Builds previous/current and current/next track pairs. |
| `pairwise_crop_cnn/dataset.py` | Loads and preprocesses thermal crops on demand. |
| `pairwise_crop_cnn/model.py` | Defines the compact CNN and metadata branch. |
| `pairwise_crop_cnn/training.py` | Data loaders, training loop, prediction, and aggregation. |
| `pairwise_crop_cnn/progress.py` | Console progress bars. |

## Quick Functionality Test

Run this before a long experiment:

```bash
python3 train_pairwise_crop_cnn.py \
  --no-tune \
  --epochs 1 \
  --max-train 32 \
  --max-val 32 \
  --max-test 32 \
  --neighbors 1 \
  --crop-size 64 \
  --batch-size 16 \
  --output-dir dist/pairwise_crop_cnn_smoke
```

This confirms that pair building, cropping, training, evaluation, and model
export work. Its metrics and `model.pt` are not meaningful final results.

## Reduced Experiment

The following command tries two learning rates with a small model:

```bash
python3 train_pairwise_crop_cnn.py \
  --epochs 4 \
  --neighbors 1 \
  --crop-size 64 \
  --batch-size 32 \
  --tune-lrs 0.001,0.0003 \
  --tune-weight-decays 0.0001 \
  --tune-base-channels 16 \
  --tune-dropouts 0.25 \
  --tune-class-weight-powers 0.5 \
  --threshold-step 0.02 \
  --output-dir dist/pairwise_crop_cnn_small
```

## Full Default Experiment

The default tuning grid contains eight candidates. A full run can be started
with:

```bash
python3 train_pairwise_crop_cnn.py \
  --epochs 8 \
  --neighbors 1 \
  --crop-size 96 \
  --batch-size 64 \
  --tune-metric macro_f1 \
  --overfit-penalty 0.15 \
  --threshold-step 0.02 \
  --output-dir dist/pairwise_crop_cnn
```

This run is computationally expensive. Every candidate repeatedly reads and
creates image crops for every epoch. A CUDA-capable GPU helps with CNN
computation, but image loading and crop preparation can remain a CPU/disk
bottleneck.

## Important Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--neighbors` | `1` | Previous and next detections used per target. |
| `--crop-size` | `96` | Final square crop size given to the CNN. |
| `--crop-scale` | `5.0` | Amount of context included around the animal box. |
| `--epochs` | `8` | Training epochs per candidate. |
| `--batch-size` | `64` | Pair samples processed in one optimization step. |
| `--num-workers` | `0` | Worker processes used for crop loading. |
| `--device` | `auto` | Uses CUDA when available, otherwise CPU. |
| `--no-tune` | off | Trains only the explicitly supplied single configuration. |
| `--tune-lrs` | `0.001,0.0003` | Learning rates included in automatic tuning. |
| `--tune-base-channels` | `16,32` | CNN widths included in automatic tuning. |
| `--tune-class-weight-powers` | `0.0,0.5` | Moving-class weighting strengths. |
| `--tune-metric` | `macro_f1` | Validation metric used to choose the model and threshold. |
| `--overfit-penalty` | `0.15` | Penalizes a large train/validation performance gap. |
| `--threshold-step` | `0.02` | Resolution of the validation threshold search. |
| `--max-train`, `--max-val`, `--max-test` | `0` | Optional stratified detection caps; `0` uses all data. |

Run `python3 train_pairwise_crop_cnn.py --help` for the complete argument list.

## Generated Outputs

Each run writes to its selected `--output-dir`:

| Output | Purpose |
| --- | --- |
| `model.pt` | Selected PyTorch state dictionary and model metadata. |
| `model_metadata.json` | Architecture, threshold, crop settings, and input descriptions. |
| `metrics.json` | Train, validation, and test metrics. |
| `tuning_results.csv` | One summary row per candidate. |
| `threshold_sweep.csv` | Validation results for every tested threshold. |
| `predictions_*.csv` | One aggregated prediction per animal detection. |

These experiment outputs remain ignored by Git. Only a genuinely selected final
weight should be explicitly allowed through `.gitignore`. Git LFS should be
considered if such a future checkpoint becomes large; the local smoke-test
checkpoint is small but is not scientifically meaningful.

## Loading A Pairwise CNN Checkpoint

A checkpoint produced by a completed run can be reconstructed as follows:

```python
import torch

from pairwise_crop_cnn.model import PairwiseCropNet

checkpoint = torch.load("dist/pairwise_crop_cnn/model.pt", map_location="cpu")
metadata = checkpoint["metadata"]
params = metadata["best_params"]

model = PairwiseCropNet(
    meta_dim=len(metadata["meta_feature_names"]),
    base_channels=int(params["base_channels"]),
    dropout=float(params["dropout"]),
)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
threshold = float(metadata["threshold"])
```

The loaded network still expects the exact three crop channels and metadata
features created by `dataset.py`. The checkpoint is not a standalone animal
detector and cannot classify arbitrary full images without bounding boxes,
track context, and pair preprocessing.

## Reproduction Notes

- Use the dependency versions from the repository `requirements.txt`.
- Run commands from the repository root so relative dataset paths resolve.
- Keep train, validation, and test flights separate during pair construction.
- The validation split selects both CNN parameters and the probability
  threshold. The test split is used only for final evaluation.
- Increase `--num-workers` carefully when disk access is fast and enough system
  memory is available.
- More pair neighbors increase training data and runtime because one animal can
  produce several pair samples.
- The random-forest baseline remains the recommended movement model for this
  project because it performed better and is far cheaper to reproduce.
