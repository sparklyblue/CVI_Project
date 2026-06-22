# Animal Detection Pipeline

## 1. Overview

This part of the project focuses on the **animal detection** stage of the full computer vision pipeline.

The overall project goal is to analyse aerial drone imagery of wildlife. The complete pipeline can be split into three main subtasks:

1. detect animals with bounding boxes
2. classify the detected animals into species
3. classify whether animals are moving or static

This README documents only the first part:

> detecting animals in thermal drone images using YOLOv8.

The detector receives a thermal image as input and returns bounding boxes around visible animals. These detections can then be used by the later species classification and movement classification parts of the project.

The final detector chosen for this part was **YOLOv8s**, because it achieved the best overall detection performance, especially in recall and mAP50-95.

---

## 2. Goal of the Detection Task

> is there an animal in the image, and where is it located?

To answer this question, the detection model was trained as a **single-class object detector**.

All animal species were merged into one class:

```text
0 Animal
```

This means that even though the original annotations contain different species labels, the detection model only learns one output category: `Animal`.

Reason for this is, because the species classification is handled separately by another project component.

---

## 3. Dataset Background

The dataset is based on the BAMBI wildlife drone dataset. The original dataset contains drone recordings of animals in natural environments and includes thermal imagery, RGB imagery, annotation labels, and tracking information.

For this detection task, the focus was on the thermal images.

The project subset contained thermal drone images from several flights. The images show animals from a top-down aerial perspective. This makes the detection task difficult because animals are often very small and can visually blend into the background.

---

## 4. Important Dataset Folders

Several label and dataset folders were used or generated during preprocessing. The most relevant ones are listed below.

### `images_thermal/`

This folder contains the thermal images used for the detection task.

The relevant substructure is:

```text
images_thermal/
└── images/
    ├── train/
    ├── val/
    └── test/
```

These folders contain the original thermal images split into train, validation, and test sets.

---

### `labels_filtered/`

This folder contains the filtered annotation labels.

These labels still contain more information than YOLO detection needs. The format is:

```text
class_id center_x center_y width height motion
```

So each line contains six values:

1. species class id
2. normalized x center
3. normalized y center
4. normalized bounding box width
5. normalized bounding box height
6. movement label

Example:

```text
3 0.411621 0.136719 0.063477 0.058594 0
```

For detection, only the bounding box information is needed. The movement column is not used by the YOLO detector.

---

### `labels_detection_only/`

This folder was generated specifically for the detection task.

The goal was to convert the original filtered labels into a single-class YOLO detection format.

Original format:

```text
species_id center_x center_y width height motion
```

Detection-only format:

```text
0 center_x center_y width height
```

Example conversion:

```text
3 0.411621 0.136719 0.063477 0.058594 0
```

becomes:

```text
0 0.411621 0.136719 0.063477 0.058594
```

The first value is changed to `0`, because all animals are treated as one class: `Animal`.

The final motion value is removed, because it belongs to the movement classification task and is not needed by YOLO.

---

### `dataset_yolo_detection/`

This is the final YOLO-ready detection dataset.

It contains the thermal images and the converted single-class labels in the structure expected by YOLOv8:

```text
dataset_yolo_detection/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

This is the main dataset folder used for YOLOv8 training, validation, and test inference.

---

### `data_yolo_detection.yaml`

This YAML file tells YOLO where the dataset is located and how many classes exist.

The relevant content is:

```yaml
path: dataset_yolo_detection

train: images/train
val: images/val
test: images/test

nc: 1

names:
  0: Animal
```

This file is important because YOLO uses it during training to find the images, labels, and class names.

---

## 5. YOLO Label Format Reminder

YOLO labels use normalized coordinates.

Each label line follows this format:

```text
class_id center_x center_y width height
```

The values are not absolute pixel positions. They are normalized between 0 and 1.

So, for example:

```text
0 0.411621 0.136719 0.063477 0.058594
```

means:

* class id: `0`, meaning Animal
* box center x-position: `0.411621` of the image width
* box center y-position: `0.136719` of the image height
* box width: `0.063477` of the image width
* box height: `0.058594` of the image height

This format was used for all generated detection labels.

---

## 6. Dataset Size

After preprocessing, the final detection dataset contained:

| Split      | Images | Labels |
| ---------- | -----: | -----: |
| Train      |  14588 |  14588 |
| Validation |   1450 |   1450 |
| Test       |   2101 |   2101 |

The number of image files and label files matched in each split, meaning every image used for training and evaluation had a corresponding label file.

The full annotation analysis counted:

| Split      | Bounding boxes |
| ---------- | -------------: |
| Train      |          31070 |
| Validation |           5416 |
| Test       |           4636 |

This means many images contain more than one animal.

---

## 7. Dataset Statistics

The detection dataset contained a total of:

```text
41122 bounding boxes
18139 labelled images
```

The average number of animals per labelled image was approximately:

```text
2.27 animals per image
```

However, the distribution is uneven. Many images contain only one animal, while some images contain many animals.

The maximum number of animals found in one image was:

```text
40 animals
```

This is relevant because crowded scenes are harder for the detector, especially when animals overlap or appear close together.

---

## 8. Why the Detection Task Is Difficult

There were some issues in regard to this dataset, for several reasons:

### Small animals

The animals are often very small compared to the full image. In many cases, the bounding boxes cover only a tiny part of the image. This makes localization difficult, especially if the animal is only a few pixels wide after resizing.

---

### Thermal image limitations

Thermal images do not contain normal colour or texture information. Animals appear as heat signatures, but so can other objects in the environment.This can cause confusion between animals and warm background regions.

---

### Background clutter

The scenes contain forests, fields, water, shadows, buildings, vegetation, and other natural structures. Some of these background regions look similar to animals in thermal imagery.

---

### Flight differences

The images come from different drone flights. This means the model has to generalize across:

* different altitudes
* different camera angles
* different backgrounds
* different lighting and thermal conditions
* different animal sizes

---

### Crowded scenes

Some images contain many animals close together. This can make it harder for YOLO to separate individual animals.

---

## 9. Model Choice

YOLOv8 was selected for this detection task.

YOLO stands for “You Only Look Once”. It is a one-stage object detection model, meaning it predicts bounding boxes and class labels directly in a single forward pass.

This makes YOLO suitable for practical detection pipelines because it is usually faster and easier to deploy than many two-stage detectors.

Two YOLOv8 variants were tested:

1. YOLOv8n
2. YOLOv8s

---

## 10. YOLOv8n vs YOLOv8s

### YOLOv8n

YOLOv8n is the nano version of YOLOv8.

It is the smallest model variant and is designed to be fast and lightweight.

Advantages:

* fastest inference
* lower memory usage
* easier to train on limited hardware
* useful for deployment on weaker machines

Disadvantages:

* lower model capacity
* may miss more difficult animals
* weaker feature representation

In this project, YOLOv8n achieved slightly higher precision, meaning it produced fewer false positives compared to YOLOv8s.

---

### YOLOv8s

YOLOv8s is the small version of YOLOv8.

It is larger than YOLOv8n and has more parameters. This gives it more capacity to learn visual patterns in the data.

Advantages:

* better feature representation
* better recall
* better localization quality
* stronger overall detection performance

Disadvantages:

* slower than YOLOv8n
* higher memory usage
* took longer to train
* more demanding on the GPU

In this project, YOLOv8s achieved better recall, mAP50, and mAP50-95. For this reason it was selected as the final detection model.

---

## 11. Training Setup

Both YOLO models were trained on the same detection dataset.

Training configuration:

| Setting      | Value                      |
| ------------ | -------------------------- |
| Model family | YOLOv8                     |
| Models       | YOLOv8n, YOLOv8s           |
| Task         | Object detection           |
| Classes      | 1                          |
| Class name   | Animal                     |
| Image size   | 1024                       |
| Epochs       | 20                         |
| Dataset YAML | `data_yolo_detection.yaml` |

The YOLOv8n model was trained with a batch size of 8.

The YOLOv8s model required a smaller batch size of 4 because it used more GPU memory and generated more heat during training.

---

## 12. Training Hardware Notes

Training was done locally using a CUDA-capable NVIDIA GPU.

The YOLOv8s run was noticeably heavier than YOLOv8n. As follow-up to the above, during the first attempts, the model caused high training time and heat load, so the training setup was adjusted by reducing the batch size.

Final YOLOv8s training used:

```text
batch = 4
```

This made the training more stable.

---

## 13. Generated Training Runs

YOLO automatically stores training results in a `runs/` directory.

For this project, the important folder is:

```text
runs/detect/runs_detection/
```

Inside this folder, there are several experiment outputs.

The important final runs are:

```text
runs/detect/runs_detection/yolov8n_baseline_1024/
runs/detect/runs_detection/yolov8s_baseline_1024/
```

There is also a debug run:

```text
runs/detect/runs_detection/baseline_debug/
```

The debug run was only used to confirm that the dataset, labels, YAML file, and GPU setup worked correctly. It was not used as the final model.

---

## 14. Contents of a YOLO Run Folder

Each YOLO training run produces several useful files.

Example:

```text
yolov8s_baseline_1024/
├── weights/
│   ├── best.pt
│   └── last.pt
├── args.yaml
└── results.csv
```

---

### `weights/best.pt`

This is the best model checkpoint selected during training.

This file should be used for inference and final evaluation.

---

### `weights/last.pt`

This is the model checkpoint from the final training epoch.

It is useful if training needs to be resumed, but it is not necessarily the best-performing checkpoint.

---

### `args.yaml`

This stores the training arguments used for the run.

It is useful for reproducibility because it records settings such as image size, batch size, model name, and other YOLO parameters.

---

### `results.csv`

This file contains the training and validation metrics for every epoch.

It was used to extract the best validation performance of each model.

---

## 15. Evaluation Metrics

The detection models were evaluated using standard object detection metrics.

---

### Precision

Precision measures how many predicted boxes are correct.

```text
Precision = True Positives / (True Positives + False Positives)
```

High precision means that when the model predicts an animal, it is more likely to actually be an animal.

In this project:

* YOLOv8n had higher precision.
* This means YOLOv8n was slightly more conservative with false positives.

---

### Recall

Recall measures how many real animals were found by the model.

```text
Recall = True Positives / (True Positives + False Negatives)
```

High recall means fewer missed animals.

In this project:

* YOLOv8s had higher recall.
* This means YOLOv8s detected more of the actual animals.

For wildlife monitoring, recall is especially important because missed animals can affect population estimation.

---

### mAP50

mAP50 means mean Average Precision at an IoU threshold of 0.50.

IoU stands for Intersection over Union and measures how much the predicted bounding box overlaps with the ground truth bounding box.

A prediction counts as correct if the predicted box overlaps the true box by at least 50%.

mAP50 is a relatively lenient detection metric.

---

### mAP50-95

mAP50-95 is stricter.

It averages mAP over multiple IoU thresholds from 0.50 to 0.95.

This means the model needs not only to detect the animal, but also to place the bounding box accurately.

In this project, mAP50-95 is the most important score for comparing localization quality.

---

## 16. Final Quantitative Results

The final validation results were:

| Model   | Precision | Recall | mAP50 | mAP50-95 |
| ------- | --------: | -----: | ----: | -------: |
| YOLOv8n |     0.558 |  0.503 | 0.431 |    0.162 |
| YOLOv8s |     0.506 |  0.567 | 0.433 |    0.177 |

---

## 17. Interpretation of Results

### YOLOv8n

YOLOv8n achieved:

```text
Precision: 0.558
Recall:    0.503
mAP50:     0.431
mAP50-95:  0.162
```

This means YOLOv8n produced slightly more reliable predictions when it decided to detect an animal.

However, its lower recall means it missed more animals compared to YOLOv8s.

---

### YOLOv8s

YOLOv8s achieved:

```text
Precision: 0.506
Recall:    0.567
mAP50:     0.433
mAP50-95:  0.177
```

YOLOv8s had slightly lower precision, but better recall and better mAP scores.

This means it detected more animals overall and localized them slightly better.

---

## 18. Why YOLOv8s Was Selected

YOLOv8s was selected as the final detection model because it achieved:

* higher recall
* higher mAP50
* higher mAP50-95
* better overall localization performance

Although YOLOv8n had slightly higher precision, YOLOv8s is more suitable for this task because the project goal is to detect animals reliably.

In wildlife monitoring, missing animals is usually more problematic than having a few additional detections.

Therefore, recall and mAP50-95 were prioritized.

---

## 19. Test Inference

After training, both models were tested on unseen test images.

A subset of 20 test images was used for qualitative comparison.

The average number of detections per image was:

| Model   | Average detections per image |
| ------- | ---------------------------: |
| YOLOv8n |                         1.45 |
| YOLOv8s |                         1.15 |

YOLOv8n produced slightly more detections on the sampled test images.

YOLOv8s produced fewer detections on average, which suggests it behaved more conservatively during test inference.

This matches the visual inspection, where YOLOv8s often produced cleaner but sometimes fewer detections.

---

## 20. Qualitative Evaluation

Visual inspection was performed using the generated prediction images.

The validation prediction images show that both YOLOv8n and YOLOv8s were able to detect animals in unseen thermal images.

The qualitative results show:

* animals were usually detected in clear thermal scenes
* detections were generally placed around visible heat signatures
* crowded scenes were more difficult
* some animals were missed when they were very small or close to background noise
* YOLOv8s was usually the stronger final choice because it had better recall and mAP scores

---

## 21. Known Weaknesses

The detection pipeline works, but it is not perfect.

### Missed small animals

The model can miss very small animals, especially if they are far from the drone or partially hidden.

---

### False positives

Some thermal background regions can be mistaken for animals. This is expected because rocks, vegetation, and other warm surfaces can look similar to animals in thermal images.

---

### Crowded groups

When many animals are close together, the model may merge them, miss some, or predict slightly inaccurate boxes.

---

### Strict localization score

The mAP50-95 scores are much lower than the mAP50 scores.

This means the model often detects the correct general area, but the exact bounding box placement is still difficult.

This makes sense because the animals are small and thermal boundaries are not always visually sharp.

---

## 22. Relation to the Full Project Pipeline

This detection module is only one part of the full system.

The full intended pipeline is:

```text
Input image
    ↓
Animal detection
    ↓
Detected bounding boxes
    ↓
Species classification
    ↓
Movement/static classification
```

However, in the final project structure, the three subtasks were mostly handled as separate models.

This detection component still provides the bounding box logic and trained detector required for the first stage of the pipeline.

---

## 23. Notebooks

The detection workflow is documented across the following notebooks.

### `1_dataset_preparation.ipynb`

Prepares the detection dataset.

Main tasks:

* checks paths and labels
* converts filtered labels to detection-only labels
* creates `labels_detection_only/`
* creates `dataset_yolo_detection/`
* creates `data_yolo_detection.yaml`
* verifies image-label counts
* visualizes sample labels

---

### `2_dataset_statistic.ipynb`

Explores dataset statistics.

Main tasks:

* counts images and bounding boxes per split
* analyses object distribution
* checks bounding box area distribution
* checks number of animals per image
* documents dataset difficulty

---

### `3_yolo_baseline.ipynb`

Trains and compares YOLOv8 models.

Main tasks:

* trains YOLOv8n
* trains YOLOv8s
* stores training runs in `runs/detect/runs_detection/`
* compares validation metrics
* selects the better model

---

### `4_test_inference.ipynb`

Runs inference on unseen test images.

Main tasks:

* loads `best.pt` weights
* selects test images
* runs YOLOv8n and YOLOv8s inference
* visualizes predictions
* compares average detections per image

---

### `5_final_evaluation.ipynb`

Summarizes final detection results.

Main tasks:

* combines validation metrics
* includes test inference findings
* compares YOLOv8n and YOLOv8s
* explains final model selection

---

## 24. How to Reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install ultralytics
pip install torch torchvision
pip install opencv-python
pip install pandas matplotlib pillow
```

---

### 2. Prepare the dataset

Run:

```text
1_dataset_preparation.ipynb
```

This creates the YOLO detection dataset and the detection YAML file.

---

### 3. Check dataset statistics

Run:

```text
2_dataset_statistic.ipynb
```

This verifies the dataset and provides basic statistics.

---

### 4. Train YOLOv8 models

Run:

```text
3_yolo_baseline.ipynb
```

This trains YOLOv8n and YOLOv8s.

The resulting model folders are saved in:

```text
runs/detect/runs_detection/
```

---

### 5. Run inference

Run:

```text
4_test_inference.ipynb
```

This loads the trained weights and visualizes predictions on test images.

---

### 6. Review final evaluation

Run:

```text
5_final_evaluation.ipynb
```

This notebook summarizes the final comparison and selected model.

---

## 25. Model Weights

The trained weights are stored in:

```text
runs/detect/runs_detection/yolov8n_baseline_1024/weights/best.pt
runs/detect/runs_detection/yolov8s_baseline_1024/weights/best.pt
```

For final inference, use:

```text
runs/detect/runs_detection/yolov8s_baseline_1024/weights/best.pt
```

because YOLOv8s was selected as the final model.

---

## 26. Final Conclusion

The animal detection task was successfully completed.

The project trained and evaluated two YOLOv8 object detection models on thermal drone imagery.

Both YOLOv8n and YOLOv8s were able to detect animals using bounding boxes. YOLOv8n was slightly more precise, but YOLOv8s achieved better recall and better mAP scores.

The final selected model is:

```text
YOLOv8s
```

because it achieved the strongest overall detection performance.

Final selected model performance:

```text
Precision: 0.506
Recall:    0.567
mAP50:     0.433
mAP50-95:  0.177
```

This detector provides a usable object detection baseline for locating animals in aerial thermal imagery and can serve as the first stage of the complete wildlife analysis pipeline.