"""NumPy logistic-regression training utilities."""

from __future__ import annotations

import math
import warnings
from collections import Counter

import numpy as np


def standardize_train_val_test(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Features are standardized from train statistics only."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std, mean, std


def sigmoid(values: np.ndarray) -> np.ndarray:
    """A numerically stable sigmoid is used by logistic regression."""
    values = np.clip(values, -50, 50)
    return 1.0 / (1.0 + np.exp(-values))


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    class_weight_power: float,
    progress_label: str,
) -> tuple[np.ndarray, float]:
    """
    A small weighted logistic regression model is trained with gradient descent.

    Class weights can be softened. A power of 0 means no class weighting, while
    a power of 1 means full balanced weighting.
    """
    n_features = x_train.shape[1]
    weights = np.zeros(n_features, dtype=np.float32)

    class_counts = Counter(int(v) for v in y_train)
    total = len(y_train)
    sample_weights = []
    for label in y_train:
        balanced_weight = total / (2 * max(1, class_counts[int(label)]))
        sample_weights.append(balanced_weight ** class_weight_power)
    sample_weights = np.asarray(sample_weights, dtype=np.float32)

    pos_rate = float(np.clip(y_train.mean(), 1e-4, 1 - 1e-4))
    bias = math.log(pos_rate / (1 - pos_rate))
    weight_sum = float(sample_weights.sum())

    for epoch in range(1, epochs + 1):
        logits = x_train @ weights + bias
        probs = sigmoid(logits)
        errors = (probs - y_train) * sample_weights
        grad_w = (x_train.T @ errors) / weight_sum + l2 * weights
        grad_b = float(errors.sum() / weight_sum)

        weights -= lr * grad_w
        bias -= lr * grad_b

        loss = weighted_log_loss(y_train, probs, sample_weights) + 0.5 * l2 * float(np.dot(weights, weights))
        bar_width = 30
        filled = int(bar_width * epoch / epochs)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(
            f"\r  {progress_label} epochs [{bar}] {epoch:4d}/{epochs} loss={loss:.4f}",
            end="",
            flush=True,
        )

    print()
    return weights, float(bias)


def weighted_log_loss(y_true: np.ndarray, probs: np.ndarray, sample_weights: np.ndarray) -> float:
    """Weighted binary cross-entropy is computed for progress reporting."""
    eps = 1e-7
    probs = np.clip(probs, eps, 1 - eps)
    losses = -(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
    return float(np.sum(losses * sample_weights) / max(float(sample_weights.sum()), eps))


def class_sample_weights(y_train: np.ndarray, class_weight_power: float) -> np.ndarray:
    """
    Per-row class weights are built for sklearn models.

    A power of 0 means no class weighting, while a power of 1 means full
    balanced weighting. Intermediate values soften the correction.
    """
    class_counts = Counter(int(v) for v in y_train)
    total = len(y_train)
    weights = []
    for label in y_train:
        balanced_weight = total / (2 * max(1, class_counts[int(label)]))
        weights.append(balanced_weight ** class_weight_power)
    return np.asarray(weights, dtype=np.float32)


def train_sklearn_classifier(
    model_name: str,
    params: dict[str, float | int | None],
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    class_weight_power: float,
):
    """A nonlinear sklearn classifier is trained on the cached feature matrix."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required for nonlinear models. "
            "Install it with: pip install -r requirements.txt"
        ) from exc

    sample_weight = class_sample_weights(y_train, class_weight_power)
    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params.get("max_features"),
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=int(params["max_iter"]),
            learning_rate=float(params["learning_rate"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            l2_regularization=float(params["l2_regularization"]),
            random_state=seed,
        )
    elif model_name == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown sklearn model: {model_name}")

    with warnings.catch_warnings():
        # This sklearn/joblib warning is emitted by some RandomForest versions
        # during parallel fitting. It is not actionable for this script.
        warnings.filterwarnings(
            "ignore",
            message="`sklearn.utils.parallel.delayed` should be used.*",
            category=UserWarning,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def sklearn_predict_moving_proba(model, x_values: np.ndarray) -> np.ndarray:
    """Moving-class probabilities are read from an sklearn classifier."""
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        moving_index = classes.index(1)
        return model.predict_proba(x_values)[:, moving_index]
    scores = model.decision_function(x_values)
    return sigmoid(scores)
