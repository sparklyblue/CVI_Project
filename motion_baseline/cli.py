"""Command-line runner for the motion baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np

from .common import (
    DEFAULT_OUTPUT_DIR,
    IMAGES_DIR,
    LABELS_DIR,
    MOTS_DIR,
    SPLITS,
    parse_float_list,
)
from .data import (
    attach_track_ids,
    build_track_index,
    load_filtered_detections,
    load_mots,
    stratified_limit,
    summarize_split,
)
from .evaluation import (
    choose_threshold,
    classification_metrics,
    evidence_group_metrics,
    make_thresholds,
    write_dict_rows,
    write_predictions,
)
from .features import FEATURE_NAMES, FEATURE_VERSION, load_or_build_feature_matrix
from .model import (
    sigmoid,
    sklearn_predict_moving_proba,
    standardize_train_val_test,
    train_logistic_regression,
    train_sklearn_classifier,
)
from .registration import ImageRegistrationCache
from .visual_debug import export_error_panels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a background-compensated motion baseline.")
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--mots-dir", type=Path, default=MOTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--neighbors", type=int, default=3, help="Number of previous and next track detections considered.")
    parser.add_argument("--max-side", type=int, default=256, help="Longest side used for image registration.")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument(
        "--model-family",
        default="both",
        choices=["logistic", "sklearn", "both"],
        help="Which model families are tried during validation tuning.",
    )
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--tune-lrs", default="0.04,0.08", help="Comma-separated learning rates to try.")
    parser.add_argument("--tune-l2s", default="0.0001,0.001,0.01", help="Comma-separated L2 values to try.")
    parser.add_argument(
        "--tune-class-weight-powers",
        default="0.0,0.5,1.0",
        help="Comma-separated class-weight strengths. 0=no weighting, 1=balanced weighting.",
    )
    parser.add_argument(
        "--tune-metric",
        default="macro_f1",
        choices=["macro_f1", "balanced_accuracy", "moving_f1", "accuracy"],
        help="Validation metric used to select the best model and threshold.",
    )
    parser.add_argument(
        "--overfit-penalty",
        type=float,
        default=0.15,
        help="Penalty applied to candidates whose train metric is higher than validation. Use 0 for pure validation selection.",
    )
    parser.add_argument("--threshold-step", type=float, default=0.02, help="Step size for validation threshold search.")
    parser.add_argument("--no-tune", action="store_true", help="Use only --lr, --l2, and full class weighting.")
    parser.add_argument(
        "--sklearn-models",
        default="random_forest,hist_gradient_boosting",
        help="Comma-separated sklearn models: random_forest,hist_gradient_boosting,gradient_boosting.",
    )
    parser.add_argument("--rf-trees", default="100,200", help="Comma-separated tree counts for RandomForest.")
    parser.add_argument("--rf-depths", default="6,8,12,16,None", help="Comma-separated max depths for RandomForest.")
    parser.add_argument("--rf-min-leaves", default="1,3,5,10", help="Comma-separated min_samples_leaf values for RandomForest.")
    parser.add_argument("--rf-max-features", default="sqrt,None", help="Comma-separated max_features values for RandomForest.")
    parser.add_argument("--hgb-iterations", default="100,200", help="Comma-separated max_iter values for HistGradientBoosting.")
    parser.add_argument("--hgb-learning-rates", default="0.05,0.1", help="Comma-separated learning rates for HistGradientBoosting.")
    parser.add_argument("--hgb-leaf-nodes", default="15,31", help="Comma-separated max_leaf_nodes values for HistGradientBoosting.")
    parser.add_argument("--hgb-l2s", default="0.0,0.1", help="Comma-separated L2 values for HistGradientBoosting.")
    parser.add_argument("--gb-trees", default="100", help="Comma-separated tree counts for GradientBoosting.")
    parser.add_argument("--gb-learning-rates", default="0.05,0.1", help="Comma-separated learning rates for GradientBoosting.")
    parser.add_argument("--gb-depths", default="2,3", help="Comma-separated max depths for GradientBoosting.")
    parser.add_argument(
        "--export-error-panels",
        type=int,
        default=20,
        help="Number of false-positive and false-negative image panels saved per split. 0 disables export.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train", type=int, default=0, help="Optional stratified cap; 0 means all train detections.")
    parser.add_argument("--max-val", type=int, default=0, help="Optional stratified cap; 0 means all val detections.")
    parser.add_argument("--max-test", type=int, default=0, help="Optional stratified cap; 0 means all test detections.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_optional_dependencies(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

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
    registration_cache = ImageRegistrationCache(max_side=args.max_side)
    cache_dir = args.feature_cache_dir or (args.output_dir / "feature_cache")
    cache_tag = feature_cache_tag(args)

    print("\nBuilding/loading train features...")
    x_train, y_train = feature_matrix_for_split(
        "train", train_dets, tracks, registration_cache, args, cache_dir, cache_tag
    )
    print("Building/loading val features...")
    x_val, y_val = feature_matrix_for_split(
        "val", val_dets, tracks, registration_cache, args, cache_dir, cache_tag
    )
    print("Building/loading test features...")
    x_test, y_test = feature_matrix_for_split(
        "test", test_dets, tracks, registration_cache, args, cache_dir, cache_tag
    )

    if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
        raise RuntimeError("Train, val, and test splits must each contain at least one detection.")

    x_train_std, x_val_std, x_test_std, feature_mean, feature_std = standardize_train_val_test(
        x_train, x_val, x_test
    )

    thresholds = make_thresholds(args.threshold_step)
    tuning_rows, threshold_rows, best = tune_all_candidates(
        thresholds=thresholds,
        args=args,
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        x_train_std=x_train_std,
        x_val_std=x_val_std,
    )

    model = best["model"]
    threshold = float(best["threshold"])
    best_params = best["params"]

    train_probs = predict_best(best, x_train, x_train_std)
    val_probs = predict_best(best, x_val, x_val_std)
    test_probs = predict_best(best, x_test, x_test_std)

    metrics = {
        "selection_metric": args.tune_metric,
        "model_family": best["family"],
        "best_params": best_params,
        "selection_score": best["score"],
        "validation_selection_score": best["val_score"],
        "train_selection_score": best["train_score"],
        "generalization_gap": best["generalization_gap"],
        "overfit_penalty": args.overfit_penalty,
        "feature_cache_tag": cache_tag,
        "feature_version": FEATURE_VERSION,
        "threshold_source": "validation_threshold_sweep",
        "feature_names": FEATURE_NAMES,
        "train": classification_metrics(y_train, train_probs, threshold),
        "val": classification_metrics(y_val, val_probs, threshold),
        "test": classification_metrics(y_test, test_probs, threshold),
        "quality_groups": {
            "train": evidence_group_metrics(x_train, y_train, train_probs, threshold, FEATURE_NAMES),
            "val": evidence_group_metrics(x_val, y_val, val_probs, threshold, FEATURE_NAMES),
            "test": evidence_group_metrics(x_test, y_test, test_probs, threshold, FEATURE_NAMES),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_dict_rows(args.output_dir / "tuning_results.csv", tuning_rows)
    write_dict_rows(args.output_dir / "threshold_sweep.csv", threshold_rows)

    write_predictions(args.output_dir / "predictions_train.csv", train_dets, train_probs, threshold)
    write_predictions(args.output_dir / "predictions_val.csv", val_dets, val_probs, threshold)
    write_predictions(args.output_dir / "predictions_test.csv", test_dets, test_probs, threshold)
    if args.export_error_panels:
        export_error_panels("train", train_dets, train_probs, threshold, tracks, args.output_dir, args.export_error_panels)
        export_error_panels("val", val_dets, val_probs, threshold, tracks, args.output_dir, args.export_error_panels)
        export_error_panels("test", test_dets, test_probs, threshold, tracks, args.output_dir, args.export_error_panels)

    save_selected_model(args.output_dir, best, model, feature_mean, feature_std, threshold)

    print(
        "\nSelected candidate: "
        f"family={best['family']}, params={best_params}, "
        f"threshold={threshold:.2f}, val_{args.tune_metric}={best['val_score']:.3f}, "
        f"train_{args.tune_metric}={best['train_score']:.3f}, "
        f"gap={best['generalization_gap']:.3f}, selection_score={best['score']:.3f}"
    )
    print_evaluation(metrics)
    print(f"\nSaved metrics, predictions, and model to: {args.output_dir}")


def ensure_optional_dependencies(args: argparse.Namespace) -> None:
    """Optional model dependencies are checked before expensive feature work starts."""
    if args.model_family in ("sklearn", "both") and importlib.util.find_spec("sklearn") is None:
        raise SystemExit(
            "scikit-learn is required for --model-family sklearn/both. "
            "Install it with: pip install -r requirements.txt\n"
            "For the old logistic-only baseline, run with: --model-family logistic"
        )


def feature_cache_tag(args: argparse.Namespace) -> str:
    """Settings that affect feature values are encoded into the cache filename."""
    parts = [
        FEATURE_VERSION,
        f"neighbors{args.neighbors}",
        f"side{args.max_side}",
        f"train{args.max_train}",
        f"val{args.max_val}",
        f"test{args.max_test}",
        f"seed{args.seed}",
    ]
    return "_".join(parts)


def feature_matrix_for_split(
    split: str,
    detections,
    tracks,
    registration_cache: ImageRegistrationCache,
    args: argparse.Namespace,
    cache_dir: Path,
    cache_tag: str,
):
    """Features are either loaded from a validated cache or freshly built."""
    if args.no_feature_cache:
        from .features import build_feature_matrix

        return build_feature_matrix(detections, tracks, registration_cache, args.neighbors)
    return load_or_build_feature_matrix(
        split=split,
        detections=detections,
        tracks=tracks,
        registration_cache=registration_cache,
        neighbors=args.neighbors,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        rebuild_cache=args.rebuild_feature_cache,
    )


def logistic_candidate_grid(args: argparse.Namespace) -> list[dict]:
    """Logistic-regression candidate dictionaries are assembled from CLI arguments."""
    if args.no_tune:
        return [{"family": "logistic", "lr": args.lr, "l2": args.l2, "class_weight_power": 1.0}]
    return [
        {"family": "logistic", "lr": lr, "l2": l2, "class_weight_power": class_weight_power}
        for lr in parse_float_list(args.tune_lrs)
        for l2 in parse_float_list(args.tune_l2s)
        for class_weight_power in parse_float_list(args.tune_class_weight_powers)
    ]


def sklearn_candidate_grid(args: argparse.Namespace) -> list[dict]:
    """Nonlinear sklearn candidate dictionaries are assembled from CLI arguments."""
    if args.no_tune:
        return []

    requested = [item.strip() for item in args.sklearn_models.split(",") if item.strip()]
    class_weight_powers = parse_float_list(args.tune_class_weight_powers)
    candidates = []
    if "random_forest" in requested:
        for n_estimators in parse_int_list(args.rf_trees):
            for max_depth in parse_optional_int_list(args.rf_depths):
                for min_samples_leaf in parse_int_list(args.rf_min_leaves):
                    for max_features in parse_optional_str_list(args.rf_max_features):
                        for class_weight_power in class_weight_powers:
                            candidates.append(
                                {
                                    "family": "sklearn",
                                    "model_name": "random_forest",
                                    "n_estimators": n_estimators,
                                    "max_depth": max_depth,
                                    "min_samples_leaf": min_samples_leaf,
                                    "max_features": max_features,
                                    "class_weight_power": class_weight_power,
                                }
                            )
    if "hist_gradient_boosting" in requested:
        for max_iter in parse_int_list(args.hgb_iterations):
            for learning_rate in parse_float_list(args.hgb_learning_rates):
                for max_leaf_nodes in parse_int_list(args.hgb_leaf_nodes):
                    for l2_regularization in parse_float_list(args.hgb_l2s):
                        for class_weight_power in class_weight_powers:
                            candidates.append(
                                {
                                    "family": "sklearn",
                                    "model_name": "hist_gradient_boosting",
                                    "max_iter": max_iter,
                                    "learning_rate": learning_rate,
                                    "max_leaf_nodes": max_leaf_nodes,
                                    "l2_regularization": l2_regularization,
                                    "class_weight_power": class_weight_power,
                                }
                            )
    if "gradient_boosting" in requested:
        for n_estimators in parse_int_list(args.gb_trees):
            for learning_rate in parse_float_list(args.gb_learning_rates):
                for max_depth in parse_int_list(args.gb_depths):
                    for class_weight_power in class_weight_powers:
                        candidates.append(
                            {
                                "family": "sklearn",
                                "model_name": "gradient_boosting",
                                "n_estimators": n_estimators,
                                "learning_rate": learning_rate,
                                "max_depth": max_depth,
                                "class_weight_power": class_weight_power,
                            }
                        )
    return candidates


def parse_int_list(raw: str) -> list[int]:
    """A comma-separated CLI value is parsed into integers."""
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_optional_int_list(raw: str) -> list[int | None]:
    """A comma-separated CLI value is parsed into integers or None."""
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(None if item.lower() == "none" else int(item))
    return values


def parse_optional_str_list(raw: str) -> list[str | None]:
    """A comma-separated CLI value is parsed into strings or None."""
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(None if item.lower() == "none" else item)
    return values


def tune_all_candidates(
    thresholds,
    args: argparse.Namespace,
    x_train,
    x_val,
    y_train,
    y_val,
    x_train_std,
    x_val_std,
):
    """Linear and nonlinear candidates are trained and scored on validation."""
    candidates = []
    if args.model_family in ("logistic", "both"):
        candidates.extend(logistic_candidate_grid(args))
    if args.model_family in ("sklearn", "both"):
        candidates.extend(sklearn_candidate_grid(args))

    tuning_rows: list[dict[str, float | int | str]] = []
    threshold_rows: list[dict[str, float | int | str]] = []
    best_score = -1.0
    best_balanced = -1.0
    best = None

    print(f"\nTraining/tuning candidates ({len(candidates)} total)...")
    for candidate_index, candidate in enumerate(candidates, start=1):
        print(f"\n  trying {format_candidate(candidate)}")
        trained = train_candidate(candidate_index, len(candidates), candidate, args, x_train, x_val, y_train, x_train_std, x_val_std)
        val_probs = trained["val_probs"]
        train_probs = trained["train_probs"]
        threshold, val_metrics, sweep_rows = choose_threshold(
            y_val,
            val_probs,
            thresholds=thresholds,
            metric_name=args.tune_metric,
        )
        train_metrics = classification_metrics(y_train, train_probs, threshold)
        val_score = float(val_metrics[args.tune_metric])
        train_score = float(train_metrics[args.tune_metric])
        generalization_gap = max(0.0, train_score - val_score)
        score = val_score - args.overfit_penalty * generalization_gap
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
                "selection_score": score,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        for sweep_row in sweep_rows:
            threshold_rows.append(
                {
                    "candidate": candidate_index,
                    **candidate,
                    **sweep_row,
                }
            )

        print(
            f"  validation {args.tune_metric}={val_score:.3f}, "
            f"train {args.tune_metric}={train_score:.3f}, "
            f"gap={generalization_gap:.3f}, selection_score={score:.3f}, "
            f"threshold={threshold:.2f}"
        )

        if score > best_score or (np.isclose(score, best_score) and balanced > best_balanced):
            best_score = score
            best_balanced = balanced
            best = {
                **trained,
                "threshold": threshold,
                "score": score,
                "val_score": val_score,
                "train_score": train_score,
                "generalization_gap": generalization_gap,
                "family": candidate["family"],
                "params": dict(candidate),
            }

    if best is None:
        raise RuntimeError("No model candidate was trained.")
    return tuning_rows, threshold_rows, best


def train_candidate(
    candidate_index: int,
    total_candidates: int,
    candidate: dict,
    args: argparse.Namespace,
    x_train,
    x_val,
    y_train,
    x_train_std,
    x_val_std,
) -> dict:
    """One candidate model is trained and returns validation probabilities."""
    if candidate["family"] == "logistic":
        weights, bias = train_logistic_regression(
            x_train_std,
            y_train,
            epochs=args.epochs,
            lr=float(candidate["lr"]),
            l2=float(candidate["l2"]),
            class_weight_power=float(candidate["class_weight_power"]),
            progress_label=f"candidate {candidate_index}/{total_candidates}",
        )
        return {
            "model": {"weights": weights, "bias": bias},
            "train_probs": sigmoid(x_train_std @ weights + bias),
            "val_probs": sigmoid(x_val_std @ weights + bias),
            "uses_standardized_features": True,
        }

    model = train_sklearn_classifier(
        model_name=str(candidate["model_name"]),
        params=candidate,
        x_train=x_train,
        y_train=y_train,
        seed=args.seed + candidate_index,
        class_weight_power=float(candidate["class_weight_power"]),
    )
    return {
        "model": model,
        "train_probs": sklearn_predict_moving_proba(model, x_train),
        "val_probs": sklearn_predict_moving_proba(model, x_val),
        "uses_standardized_features": False,
    }


def predict_best(best: dict, x_raw, x_std) -> np.ndarray:
    """The selected model is used to produce moving probabilities."""
    if best["family"] == "logistic":
        weights = best["model"]["weights"]
        bias = best["model"]["bias"]
        return sigmoid(x_std @ weights + bias)
    return sklearn_predict_moving_proba(best["model"], x_raw)


def save_selected_model(output_dir: Path, best: dict, model, feature_mean, feature_std, threshold: float) -> None:
    """The selected model is saved in a format matching its family."""
    metadata = {
        "family": best["family"],
        "params": best["params"],
        "threshold": threshold,
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if best["family"] == "logistic":
        np.savez(
            output_dir / "model.npz",
            weights=model["weights"],
            bias=np.asarray([model["bias"]], dtype=np.float32),
            feature_mean=feature_mean,
            feature_std=feature_std,
            threshold=np.asarray([threshold], dtype=np.float32),
            feature_names=np.asarray(FEATURE_NAMES),
        )
        return

    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise RuntimeError("joblib is required to save sklearn models. Install it with: pip install joblib") from exc
    joblib.dump(
        {
            "model": model,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "threshold": threshold,
            "feature_names": FEATURE_NAMES,
        },
        output_dir / "model.joblib",
    )


def format_candidate(candidate: dict) -> str:
    """A candidate dictionary is formatted for console output."""
    params = ", ".join(f"{key}={value}" for key, value in candidate.items() if key != "family")
    return f"{candidate['family']}({params})"


def print_evaluation(metrics: dict) -> None:
    """Final split metrics are printed in the same compact format as before."""
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
