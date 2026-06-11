"""Evaluation metrics and result writers."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .common import MOVING, STATIC, Detection


def make_thresholds(step: float) -> list[float]:
    """Decision thresholds are generated for validation tuning."""
    if step <= 0 or step > 0.5:
        raise ValueError("--threshold-step must be in the range (0, 0.5].")
    values = np.arange(0.05, 0.9500001, step)
    return [float(round(value, 6)) for value in values]


def choose_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: list[float],
    metric_name: str,
) -> tuple[float, dict[str, float | int], list[dict[str, float | int]]]:
    """The decision threshold is chosen by the requested validation metric."""
    best_threshold = 0.5
    best_score = -1.0
    best_balanced = -1.0
    best_metrics: dict[str, float | int] = {}
    sweep_rows = []
    for threshold in thresholds:
        metrics = classification_metrics(y_true, probs, threshold)
        sweep_rows.append(metrics)
        score = float(metrics[metric_name])
        balanced = float(metrics["balanced_accuracy"])
        if score > best_score or (math.isclose(score, best_score) and balanced > best_balanced):
            best_threshold = threshold
            best_score = score
            best_balanced = balanced
            best_metrics = metrics
    return best_threshold, best_metrics, sweep_rows


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Common binary classification metrics are computed from probabilities."""
    pred = (probs >= threshold).astype(np.int32)
    true = y_true.astype(np.int32)

    tp = int(np.sum((pred == MOVING) & (true == MOVING)))
    tn = int(np.sum((pred == STATIC) & (true == STATIC)))
    fp = int(np.sum((pred == MOVING) & (true == STATIC)))
    fn = int(np.sum((pred == STATIC) & (true == MOVING)))
    total = max(1, tp + tn + fp + fn)

    moving_precision = safe_div(tp, tp + fp)
    moving_recall = safe_div(tp, tp + fn)
    moving_f1 = safe_f1(moving_precision, moving_recall)

    static_precision = safe_div(tn, tn + fn)
    static_recall = safe_div(tn, tn + fp)
    static_f1 = safe_f1(static_precision, static_recall)

    return {
        "threshold": float(threshold),
        "samples": int(total),
        "accuracy": safe_div(tp + tn, total),
        "balanced_accuracy": (moving_recall + static_recall) / 2,
        "macro_f1": (moving_f1 + static_f1) / 2,
        "moving_precision": moving_precision,
        "moving_recall": moving_recall,
        "moving_f1": moving_f1,
        "static_precision": static_precision,
        "static_recall": static_recall,
        "static_f1": static_f1,
        "true_moving_rate": float(np.mean(true == MOVING)) if total else 0.0,
        "predicted_moving_rate": float(np.mean(pred == MOVING)) if total else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evidence_group_metrics(
    x_values: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    feature_names: list[str],
) -> dict[str, dict[str, float | int]]:
    """Metrics are computed for temporal-evidence quality groups."""
    names = {name: index for index, name in enumerate(feature_names)}
    pair_count = x_values[:, names["pair_count"]]
    has_prev = x_values[:, names["has_prev_neighbor"]]
    has_next = x_values[:, names["has_next_neighbor"]]
    max_local_ncc = x_values[:, names["max_local_registration_ncc"]]
    near_edge = x_values[:, names["near_image_edge"]]

    groups = {
        "no_temporal_neighbor": pair_count == 0,
        "one_sided_temporal_neighbor": (pair_count > 0) & ((has_prev == 0) | (has_next == 0)),
        "two_sided_good_local_context": (has_prev > 0) & (has_next > 0) & (max_local_ncc >= 0.15) & (near_edge == 0),
        "weak_local_context": (pair_count > 0) & (max_local_ncc < 0.15),
        "near_image_edge": near_edge > 0,
    }
    result = {}
    for group_name, mask in groups.items():
        if not np.any(mask):
            result[group_name] = {"samples": 0}
            continue
        result[group_name] = classification_metrics(y_true[mask], probs[mask], threshold)
    return result


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def safe_f1(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def write_predictions(
    output_path: Path,
    detections: list[Detection],
    probs: np.ndarray,
    threshold: float,
) -> None:
    """Per-detection predictions are written for later inspection."""
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "stem",
                "flight_id",
                "frame_id",
                "row_id",
                "track_id",
                "cx",
                "cy",
                "w",
                "h",
                "true_motion",
                "p_moving",
                "pred_motion",
            ],
        )
        writer.writeheader()
        for det, prob in zip(detections, probs):
            writer.writerow(
                {
                    "split": det.split,
                    "stem": det.stem,
                    "flight_id": det.flight_id,
                    "frame_id": det.frame_id,
                    "row_id": det.row_id,
                    "track_id": "" if det.track_id is None else det.track_id,
                    "cx": f"{det.cx:.6f}",
                    "cy": f"{det.cy:.6f}",
                    "w": f"{det.w:.6f}",
                    "h": f"{det.h:.6f}",
                    "true_motion": det.motion_id,
                    "p_moving": f"{float(prob):.6f}",
                    "pred_motion": int(prob >= threshold),
                }
            )


def write_dict_rows(output_path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """A list of metric dictionaries is written as a CSV file."""
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
