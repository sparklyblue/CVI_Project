"""Pair construction from recovered animal tracks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motion_baseline.common import Detection
from .progress import ProgressBar


@dataclass(frozen=True)
class PairSample:
    """
    One pairwise CNN sample.

    The target detection is the detection being classified. The neighbor is a
    previous or next detection from the same recovered track. When no neighbor
    is available, neighbor is None and the image crop is duplicated later.
    """

    target: Detection
    neighbor: Detection | None
    target_index: int
    direction: int
    frame_gap: int


def build_pair_samples(
    detections: list[Detection],
    tracks: dict[tuple[str, str, int], list[Detection]],
    neighbors: int,
    show_progress: bool = True,
) -> list[PairSample]:
    """
    Pair samples are created from nearby detections in the same recovered track.

    Multiple pair samples can point to the same target detection. During
    evaluation their probabilities are averaged back to one prediction per
    target detection.
    """
    samples: list[PairSample] = []
    progress = ProgressBar(len(detections), "building pairs") if show_progress else None
    for target_index, det in enumerate(detections):
        neighbors_for_det = track_neighbors(det, tracks, neighbors)
        if not neighbors_for_det:
            samples.append(PairSample(det, None, target_index, direction=0, frame_gap=0))
        else:
            for other in neighbors_for_det:
                direction = -1 if other.frame_id < det.frame_id else 1
                frame_gap = abs(other.frame_id - det.frame_id)
                samples.append(PairSample(det, other, target_index, direction=direction, frame_gap=frame_gap))
        if progress and (target_index + 1) % 1000 == 0:
            progress.update(target_index + 1)
    if progress:
        progress.finish(extra=f"samples={len(samples)}")
    return samples


def track_neighbors(
    det: Detection,
    tracks: dict[tuple[str, str, int], list[Detection]],
    neighbors: int,
) -> list[Detection]:
    """Nearest previous and next detections are returned in alternating order."""
    if det.track_id is None or neighbors <= 0:
        return []

    track = tracks.get((det.split, det.flight_id, det.track_id), [])
    target_index = None
    for index, candidate in enumerate(track):
        if candidate.stem == det.stem and candidate.row_id == det.row_id:
            target_index = index
            break
    if target_index is None:
        return []

    previous = list(reversed(track[max(0, target_index - neighbors) : target_index]))
    next_items = track[target_index + 1 : target_index + neighbors + 1]

    ordered = []
    for offset in range(neighbors):
        if offset < len(previous):
            ordered.append(previous[offset])
        if offset < len(next_items):
            ordered.append(next_items[offset])
    return ordered


def labels_for_detections(detections: list[Detection]) -> np.ndarray:
    """Target labels are returned in detection order."""
    return np.asarray([det.motion_id for det in detections], dtype=np.float32)


def summarize_pairs(split: str, detections: list[Detection], samples: list[PairSample]) -> None:
    """A compact pair summary is printed before training starts."""
    no_neighbor = sum(1 for sample in samples if sample.neighbor is None)
    paired_targets = len({sample.target_index for sample in samples if sample.neighbor is not None})
    print(
        f"{split}: detections={len(detections)} pair_samples={len(samples)} "
        f"paired_detections={paired_targets} no_neighbor_samples={no_neighbor}"
    )

