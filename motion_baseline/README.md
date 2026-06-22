# Motion Baseline Guide

This folder contains the current baseline for classifying each filtered animal
detection as `static` or `moving`.

The baseline is not a CNN. It is a feature-based model that tries to answer a
more careful question:

> Did this animal move relative to the local background, or did its box only
> appear to move because the drone and camera moved?

That distinction matters because the drone is moving, the frames are not always
continuous, and no flight metadata is available.

## Setup

All commands in this guide are expected to be run from the repository root.

Create a virtual environment and install the shared project requirements:

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

The thermal images are not stored in Git because of their size. Follow the
download instructions in the root `README.md`, then place the data in the
expected folders:

```txt
images_thermal/images/{train,val,test}/
labels_matched_thermal/{train,val,test}/
mots/*_gt.txt
```

Prepare the filtered labels and rebalanced splits once:

```bash
python3 build_labels.py
python3 filter_label.py
python3 rebalance_splits.py
```

The training code then expects `labels_filtered/{train,val,test}/` to exist.

## Included Trained Model

The selected random-forest weight is committed at:

```txt
dist/motion_baseline/model.joblib
```

The file is approximately 1.05 MB, so regular GitHub storage is sufficient and
Git LFS is not needed for this artifact. Other files below `dist/` are generated
reports, caches, predictions, or visualizations and remain ignored by Git.

The committed weight is the selected model from the documented run below. It
does not include the source images or feature-extraction pipeline. Input rows
must still be built in the exact order defined by `FEATURE_NAMES` in
`features.py`.

## What This Baseline Uses

The baseline uses three parts of the dataset:

| Input | Used for | Notes |
| --- | --- | --- |
| `labels_filtered/` | Training labels | This is the supervised target: `0 = static`, `1 = moving`. |
| `mots/` | Track recovery | Track ids are recovered so the same animal can be compared across nearby frames. |
| `images_thermal/images/` | Background motion features | Images are used to estimate drone/camera motion between frames. |

Species is not used as a model feature. The species id from the filtered label
is only used during track recovery, where it helps match a filtered label row
back to the correct MOTS object when several animals are present in one image.

## Important Label Format Reminder

Each label row in `labels_filtered/` has this format:

```txt
species_id cx cy width height motion_id
```

The box values are normalized YOLO values:

| Value | Meaning |
| --- | --- |
| `cx` | Box center x-position, from left to right, normalized to `0..1`. |
| `cy` | Box center y-position, from top to bottom, normalized to `0..1`. |
| `width` | Box width as a fraction of the image width. |
| `height` | Box height as a fraction of the image height. |

So `cx = 0.70` means the center of the box is 70 percent of the image width
from the left edge. `cy = 0.70` means 70 percent of the image height from the
top edge.

## Main Files

| File | Purpose |
| --- | --- |
| `train_motion_baseline.py` | Small compatibility entrypoint in the project root. It calls `motion_baseline.cli.main()`. |
| `motion_baseline/cli.py` | Main training pipeline and command-line argument handling. |
| `motion_baseline/common.py` | Shared constants, dataclasses, and argument parsing helpers. |
| `motion_baseline/data.py` | Loads labels, loads MOTS files, and recovers track ids. |
| `motion_baseline/features.py` | Builds one feature row per animal detection. Also handles feature caching. |
| `motion_baseline/registration.py` | Estimates global and local background motion from images. |
| `motion_baseline/model.py` | Trains logistic regression and sklearn models. |
| `motion_baseline/evaluation.py` | Computes metrics and writes prediction/metric CSV files. |
| `motion_baseline/visual_debug.py` | Exports visual false-positive and false-negative panels. |

## High-Level Pipeline

When the script is run, the following steps happen:

1. Filtered labels are loaded from `labels_filtered/`.
2. MOTS files are loaded from `mots/`.
3. Track ids are recovered by matching filtered boxes back to MOTS detections.
4. Detections are grouped into tracks, separately for train, validation, and test.
5. Feature rows are built or loaded from cache.
6. Candidate models are trained on the train split.
7. Candidate thresholds are tested on the validation split.
8. The best candidate is selected using validation performance and an optional overfit penalty.
9. The selected model is evaluated on train, validation, and test.
10. Metrics, predictions, model files, and optional error panels are saved.

The test split is not used for model selection. It is only used after the best
candidate and threshold have already been chosen.

## What Counts As One Training Example

One training example is one animal box in one image.

If an image contains three animals, that image contributes three rows to the
feature matrix. Each row has its own target label:

```txt
0 = static
1 = moving
```

The model does not classify a whole image as moving or static. It classifies
each detected animal separately.

## Why Track Recovery Is Needed

The filtered YOLO labels contain bounding boxes and motion labels, but they do
not contain a track id. Without a track id, it is hard to know whether an animal
in one image is the same animal as an animal in another image.

The original MOTS files do contain track ids. The baseline therefore matches
each filtered label row back to a MOTS detection from the same flight and frame.
The match uses:

- same flight id
- same frame id
- same filtered species id
- same motion id
- best bounding-box IoU

After this, nearby detections from the same recovered track can be used as
temporal evidence.

## Why Images Are Used

Using only label files gives the animal box position over time, but that is not
enough here because the drone itself moves.

For example, an animal box can move across the image even if the animal is
standing still, simply because the camera moved. The baseline therefore compares
thermal images to estimate background motion.

Two background estimates are used:

| Estimate | Meaning |
| --- | --- |
| Global background motion | Estimated from the whole downsampled image pair. |
| Local background motion | Estimated from a crop around the animal, with the animal area masked out. |

The local estimate was added because the drone motion can affect different
parts of the image differently, especially with perspective, parallax, trees,
edges, and uneven terrain.

## Feature Idea

For every detection, the feature extractor looks at nearby detections from the
same recovered track. These are called neighbor pairs.

For each neighbor pair, it measures:

| Measurement | Meaning |
| --- | --- |
| Raw motion | How far the animal box center moved in pixels. |
| Global background motion | How far the full image background appears to have moved. |
| Local background motion | How far the nearby background around the animal appears to have moved. |
| Residual motion | Raw motion minus global background motion. |
| Local residual motion | Raw motion minus local background motion. |
| Registration quality | How trustworthy the image alignment looks. |
| Frame gap | How many frame numbers are between the two detections. |

Motion values are normalized by the animal box diagonal. This keeps large and
small animals more comparable. A 10-pixel shift is not the same thing for a tiny
animal and a large animal.

## Feature Groups

The feature list is stored in `FEATURE_NAMES` in `features.py`. The most
important groups are:

| Feature group | Examples | What it tells the model |
| --- | --- | --- |
| Box location and size | `bbox_cx`, `bbox_cy`, `bbox_w`, `bbox_h`, `bbox_area` | Where the animal is and how large it is. |
| Scene context | `boxes_in_image`, `near_image_edge`, `edge_distance_norm` | Whether the case may be visually difficult. |
| Track availability | `track_available`, `track_match_iou` | Whether temporal evidence is available and how clean the match was. |
| Neighbor availability | `has_prev_neighbor`, `has_next_neighbor`, `pair_count` | Whether the animal can be compared over time. |
| Frame gaps | `mean_frame_gap`, `min_frame_gap`, `max_frame_gap` | How far apart the compared frames are. |
| Registration quality | `mean_registration_ncc`, `mean_local_registration_ncc`, `good_local_pair_count` | How reliable the background compensation may be. |
| Raw motion | `mean_raw_motion_box`, `max_raw_motion_box` | How much the box moved before camera compensation. |
| Background motion | `mean_background_motion_box`, `mean_local_background_motion_box` | How much of the movement may come from drone/camera movement. |
| Residual motion | `mean_residual_motion_box`, `mean_local_residual_motion_box` | The main evidence for animal movement. |
| Weighted motion | `weighted_local_residual_motion_box`, `weighted_raw_motion_box` | Motion summaries that give better-aligned pairs more influence. |
| Local/global disagreement | `mean_global_local_bg_disagreement_box` | Whether the global and local camera-motion estimates disagree. |

## Why There Is No Fixed Frame-Gap Rule

The script does not use a hard cutoff such as "ignore pairs more than 100
frames apart".

That is intentional. The dataset has inconsistent gaps, and different drone
flights can move at different speeds. A gap of 100 frames may be fine in one
flight and useless in another.

Instead, the baseline gives the model evidence about the pair quality:

- frame gap
- global registration quality
- local registration quality
- whether previous and next neighbors exist
- whether the animal is near the image edge

The model can then learn when temporal evidence looks useful and when it looks
weak.

## Models That Can Be Tried

The script can try two model families with several concrete model choices:

| Family | Model | Notes |
| --- | --- | --- |
| `logistic` | NumPy logistic regression | Simple, interpretable, but not very flexible. |
| `sklearn` | Random forest | Usually strong for these hand-built features. |
| `sklearn` | HistGradientBoosting | Another nonlinear tabular model. |
| `sklearn` | GradientBoosting | Available, but not included in the default sklearn list. |

The default is:

```txt
--model-family both
```

That means the script tries logistic regression and the default sklearn models.

## Recommended Run Command

After preprocessing is complete, a full run can be started with:

```bash
python3 train_motion_baseline.py \
  --model-family both \
  --epochs 800 \
  --neighbors 3 \
  --max-side 256 \
  --tune-lrs 0.02,0.04,0.08 \
  --tune-l2s 0.0001,0.001,0.01 \
  --tune-class-weight-powers 0.0,0.5,1.0 \
  --tune-metric macro_f1 \
  --overfit-penalty 0.15 \
  --threshold-step 0.01 \
  --output-dir dist/motion_baseline
```

On the first run, feature extraction may take a while because image pairs are
being compared. Later runs with the same feature settings should reuse the
feature cache.

If features must be rebuilt, add:

```bash
--rebuild-feature-cache
```

## Output Files

By default, outputs go to:

```txt
dist/motion_baseline/
```

Important files:

| Output | Meaning |
| --- | --- |
| `metrics.json` | Main evaluation report for train, validation, and test. |
| `tuning_results.csv` | One row per model candidate, including train/validation metrics. |
| `threshold_sweep.csv` | Metrics for every tested threshold for every candidate. |
| `predictions_train.csv` | Per-detection train predictions. |
| `predictions_val.csv` | Per-detection validation predictions. |
| `predictions_test.csv` | Per-detection test predictions. |
| `model_metadata.json` | Selected model family, parameters, threshold, feature version, and feature names. |
| `model.joblib` | Saved sklearn model. The selected baseline weight is committed to the repository. |
| `model.npz` | Saved logistic regression model, if the selected model is logistic regression. |
| `feature_cache/` | Cached feature matrices. These are generated files and should not be committed. |
| `error_panels/` | Optional visual inspection panels for false positives and false negatives. |

## Loading the Trained Model

Only load `joblib` files from trusted sources, because the format can execute
Python code while loading. The repository artifact can be opened as follows:

```python
from pathlib import Path

import joblib

artifact = joblib.load(Path("dist/motion_baseline/model.joblib"))
model = artifact["model"]
threshold = float(artifact["threshold"])
feature_names = artifact["feature_names"]

# x_features must contain the features produced by features.py in feature_names order.
moving_index = list(model.classes_).index(1)
p_moving = model.predict_proba(x_features)[:, moving_index]
predicted_motion = (p_moving >= threshold).astype(int)
```

For an end-to-end reproduction, use `train_motion_baseline.py`; it performs
track recovery, feature extraction, model tuning, evaluation, and export. The
saved model alone cannot turn arbitrary images into detections because animal
boxes and temporal track context are required first.

## Feature Cache

Feature building is the slowest part of the baseline because every detection
may compare image pairs globally and locally. The cache saves the resulting
feature matrices so model tuning can be repeated quickly.

The cache is only reused when:

- the split is the same
- the feature version is the same
- `--neighbors` is the same
- `--max-side` is the same
- debug caps such as `--max-train` are the same
- the detection row keys match in the same order
- the feature names match

This protects against accidentally training on stale feature rows.

Use:

```bash
--rebuild-feature-cache
```

when feature code changes or when you want to force a clean rebuild.

Use:

```bash
--no-feature-cache
```

only for debugging, because it recomputes features every run.

## Argument Reference

### Data and Output Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--images-dir` | `images_thermal/images` | Folder containing `train`, `val`, and `test` image folders. |
| `--labels-dir` | `labels_filtered` | Folder containing filtered YOLO labels. |
| `--mots-dir` | `mots` | Folder containing raw MOTS tracking files. |
| `--output-dir` | `dist/motion_baseline` | Where metrics, predictions, panels, model files, and cache are saved. |
| `--feature-cache-dir` | none | Optional custom cache folder. If omitted, cache is stored under `output-dir/feature_cache`. |
| `--rebuild-feature-cache` | off | Forces feature matrices to be rebuilt even if matching cache files exist. |
| `--no-feature-cache` | off | Disables cache loading and saving. |

### Feature Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--neighbors` | `3` | Number of previous and next same-track detections to compare with each detection. |
| `--max-side` | `256` | Longest side used when resizing images for registration. Larger can be more accurate but slower. |

### Model Selection Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--model-family` | `both` | Which model families to try: `logistic`, `sklearn`, or `both`. |
| `--no-tune` | off | Uses only the single logistic settings from `--lr` and `--l2`. Sklearn candidates are skipped. |
| `--tune-metric` | `macro_f1` | Validation metric used to pick the best candidate and threshold. |
| `--overfit-penalty` | `0.15` | Penalizes candidates whose train score is much higher than validation score. |
| `--threshold-step` | `0.02` | Step size for threshold search from about `0.05` to `0.95`. Smaller is slower but more precise. |
| `--seed` | `7` | Random seed for repeatability. |

The selection score is:

```txt
selection_score = validation_metric - overfit_penalty * max(0, train_metric - validation_metric)
```

With `--overfit-penalty 0`, the script selects purely by validation metric.

### Logistic Regression Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--epochs` | `800` | Number of gradient-descent epochs for logistic regression. |
| `--lr` | `0.08` | Learning rate used when `--no-tune` is enabled. |
| `--l2` | `0.001` | L2 regularization used when `--no-tune` is enabled. |
| `--tune-lrs` | `0.04,0.08` | Learning rates tried during tuning. |
| `--tune-l2s` | `0.0001,0.001,0.01` | L2 values tried during tuning. |
| `--tune-class-weight-powers` | `0.0,0.5,1.0` | Class-weight strengths tried during tuning. |

`class_weight_power` controls how strongly the smaller moving class is weighted:

| Value | Meaning |
| --- | --- |
| `0.0` | No class weighting. |
| `0.5` | Soft class weighting. |
| `1.0` | Full balanced class weighting. |

### Sklearn Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--sklearn-models` | `random_forest,hist_gradient_boosting` | Which sklearn models are tried. |
| `--rf-trees` | `100,200` | Random forest tree counts. |
| `--rf-depths` | `6,8,12,16,None` | Random forest max depths. `None` means unlimited depth. |
| `--rf-min-leaves` | `1,3,5,10` | Random forest minimum samples per leaf. Higher values reduce overfitting. |
| `--rf-max-features` | `sqrt,None` | Number of features considered per split. |
| `--hgb-iterations` | `100,200` | HistGradientBoosting iteration counts. |
| `--hgb-learning-rates` | `0.05,0.1` | HistGradientBoosting learning rates. |
| `--hgb-leaf-nodes` | `15,31` | HistGradientBoosting leaf-node limits. |
| `--hgb-l2s` | `0.0,0.1` | HistGradientBoosting L2 regularization values. |
| `--gb-trees` | `100` | GradientBoosting tree counts. |
| `--gb-learning-rates` | `0.05,0.1` | GradientBoosting learning rates. |
| `--gb-depths` | `2,3` | GradientBoosting max depths. |

### Debug Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--max-train` | `0` | Optional stratified cap for train detections. `0` means all detections. |
| `--max-val` | `0` | Optional stratified cap for validation detections. |
| `--max-test` | `0` | Optional stratified cap for test detections. |
| `--export-error-panels` | `20` | Number of false-positive and false-negative panels saved per split. `0` disables export. |

The `--max-*` arguments are mainly for smoke tests and quick debugging. They
should not be used for final reported metrics.

## How Threshold Selection Works

Most classifiers output a probability, not a hard class. Here that probability
means:

```txt
p_moving = model estimate that this detection is moving
```

A threshold turns that probability into a label:

```txt
if p_moving >= threshold:
    prediction = moving
else:
    prediction = static
```

The script tries many thresholds on the validation set and keeps the threshold
that gives the best selected validation metric. This is important because the
moving class is smaller, and a default `0.50` threshold is not always the best
choice.

## Metrics Explained

The final console output looks like this:

```txt
train: accuracy=... balanced_acc=... moving_f1=... macro_f1=... confusion(tn,fp,fn,tp)=(...)
val:   accuracy=... balanced_acc=... moving_f1=... macro_f1=... confusion(tn,fp,fn,tp)=(...)
test:  accuracy=... balanced_acc=... moving_f1=... macro_f1=... confusion(tn,fp,fn,tp)=(...)
```

### Confusion Matrix

The confusion tuple is:

```txt
(tn, fp, fn, tp)
```

| Term | Meaning |
| --- | --- |
| `tn` | True negatives: static animals correctly predicted as static. |
| `fp` | False positives: static animals incorrectly predicted as moving. |
| `fn` | False negatives: moving animals incorrectly predicted as static. |
| `tp` | True positives: moving animals correctly predicted as moving. |

For this task:

- false positives mean the model is too eager to call motion
- false negatives mean the model misses actual movement

### Accuracy

Accuracy is the total fraction of correct predictions:

```txt
(tp + tn) / all detections
```

Accuracy can look good even when moving animals are missed, because static
detections are more common. It should not be the only metric used here.

### Moving Precision

Moving precision answers:

> Of the detections predicted as moving, how many were truly moving?

High precision means fewer false alarms.

### Moving Recall

Moving recall answers:

> Of the truly moving detections, how many did the model find?

High recall means fewer missed moving animals.

### Moving F1

Moving F1 combines moving precision and moving recall. It is useful when the
main interest is the moving class specifically.

### Balanced Accuracy

Balanced accuracy averages recall for the static class and recall for the
moving class.

This is useful because the dataset is imbalanced. A model cannot get a great
balanced accuracy by simply predicting the majority class all the time.

### Macro F1

Macro F1 averages the F1 score for static and moving.

This is the default tuning metric because it gives both classes importance. It
is usually more honest than accuracy for this dataset.

## Quality Group Metrics

`metrics.json` also contains `quality_groups`. These are diagnostic groups used
to understand where the model works or fails.

| Group | Meaning |
| --- | --- |
| `no_temporal_neighbor` | No same-track neighbor was available. |
| `one_sided_temporal_neighbor` | Only previous or only next context was available. |
| `two_sided_good_local_context` | Previous and next context exist and local registration looks usable. |
| `weak_local_context` | Temporal neighbors exist, but local registration quality is weak. |
| `near_image_edge` | The animal is close to an image edge. |

These metrics are not separate models. They are slices of the same predictions,
useful for explaining failure modes.

## How To Read The Current Kind Of Result

A recent strong baseline run selected a random forest like this:

```txt
family=sklearn
model_name=random_forest
n_estimators=100
max_depth=6
min_samples_leaf=3
max_features=sqrt
class_weight_power=0.0
threshold=0.38
```

The recorded evaluation for this selected model was:

| Split | Accuracy | Balanced accuracy | Moving F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.895 | 0.829 | 0.776 | 0.854 |
| Validation | 0.782 | 0.555 | 0.226 | 0.549 |
| Test | 0.907 | 0.811 | 0.647 | 0.797 |

The test confusion matrix was `TN=3878`, `FP=251`, `FN=188`, and
`TP=403`. Accuracy is high partly because static detections are more common, so
balanced accuracy, moving F1, and macro F1 should also be reported.

The important pattern was:

- train performance was good but not perfectly memorized
- validation performance was modest
- test performance was much stronger than validation

That suggests the validation split contains some unusually difficult flights,
not just that the model is useless. Earlier inspection showed validation errors
were heavily concentrated in specific flights, especially cases with weak local
context, large background shifts, tiny animals, and missing neighbors.

The honest interpretation is:

> The baseline is useful and defensible, but its reliability depends strongly
> on flight conditions and temporal evidence quality.

## Strengths

This baseline has several useful strengths:

- It directly addresses drone/camera motion instead of treating box movement as animal movement.
- It does not rely on species as a shortcut.
- It is interpretable because the features have physical meanings.
- It can be tuned automatically across model parameters and thresholds.
- It exports per-detection predictions for later analysis.
- It exports visual error panels, which makes failure inspection much easier.
- It creates a reproducible baseline before trying a CNN.

## Weaknesses

The main weaknesses come from missing information and imperfect temporal data:

- Drone metadata is missing. There is no altitude, speed, camera angle, GPS,
  IMU, lens, or pose information.
- Frame gaps are irregular. Some neighboring labels may be close in time,
  others may be separated by many frames.
- Track recovery depends on matching filtered labels back to MOTS boxes.
- Image registration assumes mostly translational motion, which is an
  approximation.
- Local registration can fail near image edges or in visually weak crops.
- Tiny animals are difficult because small localization errors become large
  relative motion errors.
- Static animals can look moving when the background compensation is poor.
- Moving animals can look static when only one weak temporal neighbor exists.

These limitations are important to report. They explain why the baseline should
not be expected to reach perfect performance.

## Why A CNN Might Help Later

A CNN or video model could learn visual cues this baseline does not model
directly, such as:

- animal posture
- blur or heat-shape changes
- local texture movement
- more complex camera motion
- scene type
- motion patterns across small image crops

However, a CNN will also have to deal with the same hard facts:

- irregular frame gaps
- moving drone footage
- missing flight metadata
- small animals
- split-specific flight differences

That is why this baseline is useful first. It gives a clear, interpretable
reference point that a CNN should beat.

## Cross-Validation Note

The current script uses the provided train/validation/test splits.

Cross-validation could be added later, but it should be grouped by flight id.
Randomly splitting individual detections would leak similar frames from the
same flight into different folds and make the results too optimistic.

A careful future setup would be:

```txt
GroupKFold over flight_id on train + validation
select parameters from grouped CV
retrain on train + validation
evaluate once on test
```

The current baseline is still fine as a pushable project milestone because the
test split remains untouched until final evaluation.

## Reproduction Notes

- Use the dependency versions in the repository `requirements.txt`, especially
  the recorded scikit-learn version, when loading the committed `joblib` file.
- Run commands from the repository root so the default relative paths resolve.
- The first feature build is slow because image registration is performed for
  many temporal pairs. Matching feature caches make later tuning runs faster.
- Use `--rebuild-feature-cache` after changing feature extraction, neighbor
  count, image-registration size, or the underlying labels.
- Keep train, validation, and test flights separate. Test data must not be used
  for parameter or threshold selection.
- The result is a per-animal, per-image motion classification. It is not a
  whole-image classifier or an animal detector.
