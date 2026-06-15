"""Training and prediction helpers for the pairwise crop CNN."""

from __future__ import annotations

import random
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .progress import ProgressBar


def set_torch_seed(seed: int) -> None:
    """Random seeds are set for repeatable candidate training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(raw_device: str) -> torch.device:
    """The requested device is converted to a torch device."""
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    """A DataLoader is created with deterministic shuffling when requested."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def positive_class_weight(labels: np.ndarray, class_weight_power: float) -> torch.Tensor:
    """The moving class can be upweighted to compensate for class imbalance."""
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    if positives <= 0 or negatives <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor((negatives / positives) ** class_weight_power, dtype=torch.float32)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    labels: np.ndarray,
    lr: float,
    weight_decay: float,
    class_weight_power: float,
    epochs: int,
    device: torch.device,
    candidate_label: str,
) -> nn.Module:
    """One CNN candidate is trained for the requested number of epochs."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = positive_class_weight(labels, class_weight_power).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = ProgressBar(len(train_loader), f"{candidate_label} epoch {epoch}/{epochs}")
        for batch_index, batch in enumerate(train_loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(image, meta)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            batch_size = int(label.shape[0])
            running_loss += float(loss.detach().cpu()) * batch_size
            seen += batch_size
            progress.update(batch_index, extra=f"loss={running_loss / max(1, seen):.4f}")
        progress.finish(extra=f"loss={running_loss / max(1, seen):.4f}")
    return model


def predict_samples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Moving probabilities are predicted for pair samples."""
    model.eval()
    probs: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    progress = ProgressBar(len(loader), label)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            logits = model(image, meta)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(batch_probs)
            indices.append(batch["target_index"].numpy())
            progress.update(batch_index)
    progress.finish()
    return np.concatenate(indices), np.concatenate(probs)


def aggregate_detection_probs(
    target_indices: np.ndarray,
    sample_probs: np.ndarray,
    detection_count: int,
) -> np.ndarray:
    """Pair probabilities are averaged back to one probability per detection."""
    sums = np.zeros(detection_count, dtype=np.float64)
    counts = np.zeros(detection_count, dtype=np.float64)
    for index, prob in zip(target_indices, sample_probs):
        sums[int(index)] += float(prob)
        counts[int(index)] += 1.0
    return np.divide(sums, counts, out=np.full(detection_count, 0.5), where=counts > 0).astype(np.float32)


def state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """A model state dict is copied to CPU so it can be kept after tuning."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def copy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A state dict is deep-copied before it is stored as the best candidate."""
    return deepcopy(state_dict)

