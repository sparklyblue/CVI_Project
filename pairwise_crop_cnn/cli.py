"""Command-line runner for the pairwise crop CNN."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from motion_baseline.common import (
    IMAGES_DIR,
    LABELS_DIR,
    MOTS_DIR,
    SPLITS,
    parse_float_list,
    parse_int_list,
)
from motion_baseline.data import (
    attach_track_ids,
    build_track_index,
    load_filtered_detections,
    load_mots,
    stratified_limit,
    summarize_split,
)
from motion_baseline.evaluation import (
    choose_threshold,
    classification_metrics,
    make_thresholds,
    write_dict_rows,
    write_predictions,
)
from .pairs import build_pair_samples, labels_for_detections, summarize_pairs


DEFAULT_OUTPUT_DIR = Path("dist/pairwise_crop_cnn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pairwise crop CNN for static/moving classification.")
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--mots-dir", type=Path, default=MOTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--neighbors", type=int, default=1, help="Previous/next same-track detections used per target.")
    parser.add_argument("--crop-size", type=int, default=96, help="Final square crop size used by the CNN.")
    parser.add_argument("--crop-scale", type=float, default=5.0, help="Context crop size relative to the target box.")
    parser.add_argument("--min-crop-pixels", type=int, default=64)
    parser.add_argument("--max-crop-pixels", type=int, default=512)
    parser.add_argument("--frame-gap-scale", type=float, default=2000.0)
    parser.add_argument("--max-motion-box", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--no-tune", action="store_true", help="Use only the single explicit CNN setting.")
    parser.add_argument("--tune-lrs", default="0.001,0.0003")
    parser.add_argument("--tune-weight-decays", default="0.0001")
    parser.add_argument("--tune-base-channels", default="16,32")
    parser.add_argument("--tune-dropouts", default="0.25")
    parser.add_argument("--tune-class-weight-powers", default="0.0,0.5")
    parser.add_argument(
        "--tune-metric",
        default="macro_f1",
        choices=["macro_f1", "balanced_accuracy", "moving_f1", "accuracy"],
        help="Validation metric used to select the best candidate and threshold.",
    )
    parser.add_argument("--overfit-penalty", type=float, default=0.15)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_torch_dependency()

    from .dataset import META_FEATURE_NAMES, CropSettings, PairCropDataset
    from .model import PairwiseCropNet
    from .training import (
        aggregate_detection_probs,
        choose_device,
        copy_state_dict,
        make_loader,
        predict_samples,
        set_torch_seed,
        state_dict_to_cpu,
        train_model,
    )
    import torch

    set_torch_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    print("Loading filtered labels...")
    detections_by_split = load_filtered_detections(args.images_dir, args.labels_dir)

    print("Loading MOTS tracks...")
    mots = load_mots(args.mots_dir)

    print("Recovering track ids...")
    attach_track_ids(detections_by_split, mots)

    train_dets = stratified_limit(detections_by_split["train"], args.max_train, args.seed)
    val_dets = stratified_limit(detections_by_split["val"], args.max_val, args.seed + 1)
    test_dets = stratified_limit(detections_by_split["test"], args.max_test, args.seed + 2)
    limited_by_split = {"train": train_dets, "val": val_dets, "test": test_dets}

    print("\nSplit summary:")
    for split in SPLITS:
        summarize_split(split, limited_by_split[split])

    tracks = build_track_index(limited_by_split)
    print("\nBuilding pair samples...")
    train_samples = build_pair_samples(train_dets, tracks, args.neighbors)
    val_samples = build_pair_samples(val_dets, tracks, args.neighbors)
    test_samples = build_pair_samples(test_dets, tracks, args.neighbors)
    summarize_pairs("train", train_dets, train_samples)
    summarize_pairs("val", val_dets, val_samples)
    summarize_pairs("test", test_dets, test_samples)

    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("Train, val, and test must each contain at least one pair sample.")

    settings = CropSettings(
        crop_size=args.crop_size,
        crop_scale=args.crop_scale,
        min_crop_pixels=args.min_crop_pixels,
        max_crop_pixels=args.max_crop_pixels,
        frame_gap_scale=args.frame_gap_scale,
        max_motion_box=args.max_motion_box,
    )
    train_dataset = PairCropDataset(train_samples, settings)
    val_dataset = PairCropDataset(val_samples, settings)
    test_dataset = PairCropDataset(test_samples, settings)

    y_train = labels_for_detections(train_dets)
    y_val = labels_for_detections(val_dets)
    y_test = labels_for_detections(test_dets)
    sample_train_labels = np.asarray([sample.target.motion_id for sample in train_samples], dtype=np.float32)

    candidates = candidate_grid(args)
    thresholds = make_thresholds(args.threshold_step)
    tuning_rows: list[dict[str, float | int | str]] = []
    threshold_rows: list[dict[str, float | int | str]] = []
    best = None
    best_score = -1.0
    best_balanced = -1.0

    print(f"\nTraining/tuning CNN candidates ({len(candidates)} total)...")
    for candidate_index, candidate in enumerate(candidates, start=1):
        set_torch_seed(args.seed + candidate_index)
        print(f"\n  trying {format_candidate(candidate)}")

        train_loader = make_loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + candidate_index,
            num_workers=args.num_workers,
        )
        eval_train_loader = make_loader(train_dataset, args.batch_size, False, args.seed, args.num_workers)
        val_loader = make_loader(val_dataset, args.batch_size, False, args.seed, args.num_workers)

        model = PairwiseCropNet(
            meta_dim=len(META_FEATURE_NAMES),
            base_channels=int(candidate["base_channels"]),
            dropout=float(candidate["dropout"]),
        )
        model = train_model(
            model=model,
            train_loader=train_loader,
            labels=sample_train_labels,
            lr=float(candidate["lr"]),
            weight_decay=float(candidate["weight_decay"]),
            class_weight_power=float(candidate["class_weight_power"]),
            epochs=args.epochs,
            device=device,
            candidate_label=f"candidate {candidate_index}/{len(candidates)}",
        )

        train_indices, train_pair_probs = predict_samples(model, eval_train_loader, device, "train predict")
        val_indices, val_pair_probs = predict_samples(model, val_loader, device, "val predict")
        train_probs = aggregate_detection_probs(train_indices, train_pair_probs, len(train_dets))
        val_probs = aggregate_detection_probs(val_indices, val_pair_probs, len(val_dets))

        threshold, val_metrics, sweep_rows = choose_threshold(y_val, val_probs, thresholds, args.tune_metric)
        train_metrics = classification_metrics(y_train, train_probs, threshold)
        val_score = float(val_metrics[args.tune_metric])
        train_score = float(train_metrics[args.tune_metric])
        generalization_gap = max(0.0, train_score - val_score)
        selection_score = val_score - args.overfit_penalty * generalization_gap
        balanced = float(val_metrics["balanced_accuracy"])

        tuning_rows.append(
            {
                "candidate": candidate_index,
                **candidate,
                "best_threshold": threshold,
                "train_selection_metric": train_score,
                "val_selection_metric": val_score,
                "generalization_gap": generalization_gap,
                "overfit_penalty": args.overfit_penalty,
                "selection_score": selection_score,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        for sweep_row in sweep_rows:
            threshold_rows.append({"candidate": candidate_index, **candidate, **sweep_row})

        print(
            f"  validation {args.tune_metric}={val_score:.3f}, "
            f"train {args.tune_metric}={train_score:.3f}, "
            f"gap={generalization_gap:.3f}, selection_score={selection_score:.3f}, "
            f"threshold={threshold:.2f}"
        )

        if selection_score > best_score or (np.isclose(selection_score, best_score) and balanced > best_balanced):
            best_score = selection_score
            best_balanced = balanced
            best = {
                "candidate": candidate_index,
                "params": dict(candidate),
                "threshold": threshold,
                "score": selection_score,
                "val_score": val_score,
                "train_score": train_score,
                "generalization_gap": generalization_gap,
                "state_dict": copy_state_dict(state_dict_to_cpu(model)),
            }

    if best is None:
        raise RuntimeError("No CNN candidate was trained.")

    best_model = PairwiseCropNet(
        meta_dim=len(META_FEATURE_NAMES),
        base_channels=int(best["params"]["base_channels"]),
        dropout=float(best["params"]["dropout"]),
    )
    best_model.load_state_dict(best["state_dict"])
    best_model.to(device)

    train_probs = predict_detection_probs(best_model, train_dataset, len(train_dets), args, device, "best train predict")
    val_probs = predict_detection_probs(best_model, val_dataset, len(val_dets), args, device, "best val predict")
    test_probs = predict_detection_probs(best_model, test_dataset, len(test_dets), args, device, "best test predict")
    threshold = float(best["threshold"])

    metrics = {
        "model": "pairwise_crop_cnn",
        "selection_metric": args.tune_metric,
        "best_params": best["params"],
        "threshold": threshold,
        "selection_score": best["score"],
        "validation_selection_score": best["val_score"],
        "train_selection_score": best["train_score"],
        "generalization_gap": best["generalization_gap"],
        "overfit_penalty": args.overfit_penalty,
        "crop_settings": settings.__dict__,
        "meta_feature_names": META_FEATURE_NAMES,
        "pair_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "train": classification_metrics(y_train, train_probs, threshold),
        "val": classification_metrics(y_val, val_probs, threshold),
        "test": classification_metrics(y_test, test_probs, threshold),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_dict_rows(args.output_dir / "tuning_results.csv", tuning_rows)
    write_dict_rows(args.output_dir / "threshold_sweep.csv", threshold_rows)
    write_predictions(args.output_dir / "predictions_train.csv", train_dets, train_probs, threshold)
    write_predictions(args.output_dir / "predictions_val.csv", val_dets, val_probs, threshold)
    write_predictions(args.output_dir / "predictions_test.csv", test_dets, test_probs, threshold)

    metadata = {
        "model": "pairwise_crop_cnn",
        "best_params": best["params"],
        "threshold": threshold,
        "crop_settings": settings.__dict__,
        "meta_feature_names": META_FEATURE_NAMES,
        "input_channels": ["target_crop", "neighbor_crop", "absolute_difference"],
    }
    (args.output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.save(
        {
            "state_dict": best_model.cpu().state_dict(),
            "metadata": metadata,
        },
        args.output_dir / "model.pt",
    )

    print(
        "\nSelected candidate: "
        f"params={best['params']}, threshold={threshold:.2f}, "
        f"val_{args.tune_metric}={best['val_score']:.3f}, "
        f"train_{args.tune_metric}={best['train_score']:.3f}, "
        f"gap={best['generalization_gap']:.3f}, selection_score={best['score']:.3f}"
    )
    print_evaluation(metrics)
    print(f"\nSaved metrics, predictions, and model to: {args.output_dir}")


def ensure_torch_dependency() -> None:
    """PyTorch is checked before the expensive data-loading work starts."""
    if importlib.util.find_spec("torch") is None:
        raise SystemExit(
            "PyTorch is required for the pairwise crop CNN. "
            "Install dependencies with: pip install -r requirements.txt"
        )


def candidate_grid(args: argparse.Namespace) -> list[dict[str, float | int]]:
    """CNN hyperparameter candidates are assembled from CLI values."""
    if args.no_tune:
        return [
            {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "base_channels": args.base_channels,
                "dropout": args.dropout,
                "class_weight_power": args.class_weight_power,
            }
        ]
    return [
        {
            "lr": lr,
            "weight_decay": weight_decay,
            "base_channels": base_channels,
            "dropout": dropout,
            "class_weight_power": class_weight_power,
        }
        for lr in parse_float_list(args.tune_lrs)
        for weight_decay in parse_float_list(args.tune_weight_decays)
        for base_channels in parse_int_list(args.tune_base_channels)
        for dropout in parse_float_list(args.tune_dropouts)
        for class_weight_power in parse_float_list(args.tune_class_weight_powers)
    ]


def predict_detection_probs(model, dataset, detection_count: int, args: argparse.Namespace, device, label: str) -> np.ndarray:
    """Pair sample predictions are aggregated to one probability per detection."""
    from .training import aggregate_detection_probs, make_loader, predict_samples

    loader = make_loader(dataset, args.batch_size, False, args.seed, args.num_workers)
    indices, pair_probs = predict_samples(model, loader, device, label)
    return aggregate_detection_probs(indices, pair_probs, detection_count)


def format_candidate(candidate: dict[str, float | int]) -> str:
    """A candidate dictionary is formatted for console output."""
    params = ", ".join(f"{key}={value}" for key, value in candidate.items())
    return f"pairwise_crop_cnn({params})"


def print_evaluation(metrics: dict) -> None:
    """Final split metrics are printed in the familiar compact format."""
    print("\nEvaluation:")
    for split in SPLITS:
        split_metrics = metrics[split]
        print(
            f"{split}: "
            f"accuracy={split_metrics['accuracy']:.3f} "
            f"balanced_acc={split_metrics['balanced_accuracy']:.3f} "
            f"moving_f1={split_metrics['moving_f1']:.3f} "
            f"macro_f1={split_metrics['macro_f1']:.3f} "
            f"confusion(tn,fp,fn,tp)="
            f"({split_metrics['tn']},{split_metrics['fp']},"
            f"{split_metrics['fn']},{split_metrics['tp']})"
        )
