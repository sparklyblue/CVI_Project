"""Feature extraction and feature-matrix caching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import Detection, detection_cache_key
from .registration import ImageRegistrationCache


@dataclass
class PairMotion:
    """Motion features between one target detection and one track neighbor."""

    frame_gap: float
    raw_dx: float
    raw_dy: float
    bg_dx: float
    bg_dy: float
    local_bg_dx: float
    local_bg_dy: float
    residual_dx: float
    residual_dy: float
    local_residual_dx: float
    local_residual_dy: float
    box_diag: float
    ncc: float
    peak_ratio: float
    local_ncc: float
    local_peak_ratio: float


FEATURE_VERSION = "v2_local_background"

FEATURE_NAMES = [
    "bbox_cx",
    "bbox_cy",
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "bbox_aspect",
    "boxes_in_image",
    "track_available",
    "track_match_iou",
    "edge_distance_norm",
    "near_image_edge",
    "has_prev_neighbor",
    "has_next_neighbor",
    "pair_count",
    "prev_pair_count",
    "next_pair_count",
    "mean_frame_gap",
    "min_frame_gap",
    "max_frame_gap",
    "mean_registration_ncc",
    "max_registration_ncc",
    "mean_local_registration_ncc",
    "max_local_registration_ncc",
    "mean_peak_ratio_log",
    "max_peak_ratio_log",
    "mean_local_peak_ratio_log",
    "max_local_peak_ratio_log",
    "mean_raw_motion_box",
    "max_raw_motion_box",
    "mean_background_motion_box",
    "max_background_motion_box",
    "mean_local_background_motion_box",
    "max_local_background_motion_box",
    "mean_residual_motion_box",
    "max_residual_motion_box",
    "mean_local_residual_motion_box",
    "max_local_residual_motion_box",
    "weighted_residual_motion_box",
    "weighted_local_residual_motion_box",
    "weighted_raw_motion_box",
    "weighted_background_motion_box",
    "weighted_local_background_motion_box",
    "mean_global_local_bg_disagreement_box",
    "max_global_local_bg_disagreement_box",
    "mean_abs_residual_x_box",
    "mean_abs_residual_y_box",
    "mean_abs_local_residual_x_box",
    "mean_abs_local_residual_y_box",
    "good_global_pair_count",
    "good_local_pair_count",
    "temporal_quality_score",
]


def neighbor_candidates(
    det: Detection,
    tracks: dict[tuple[str, str, int], list[Detection]],
    neighbors: int,
) -> list[Detection]:
    """
    Previous and next detections from the same track are selected by track order.

    No fixed frame-gap cutoff is used. Image registration quality is later
    exposed to the model so visually poor pairs can be treated as weak evidence.
    """
    if det.track_id is None:
        return []
    track = tracks.get((det.split, det.flight_id, det.track_id), [])
    if not track:
        return []

    target_index = None
    for idx, candidate in enumerate(track):
        if candidate.stem == det.stem and candidate.row_id == det.row_id:
            target_index = idx
            break
    if target_index is None:
        return []

    start = max(0, target_index - neighbors)
    end = min(len(track), target_index + neighbors + 1)
    return [item for idx, item in enumerate(track[start:end], start=start) if idx != target_index]


def pair_motion(det: Detection, other: Detection, registration_cache: ImageRegistrationCache) -> PairMotion:
    """One neighbor pair is converted into camera-compensated motion values."""
    bg_dx, bg_dy, ncc, peak_ratio = registration_cache.register(det.image_path, other.image_path)
    crop_box = context_crop_box(det)
    local_bg_dx, local_bg_dy, local_ncc, local_peak_ratio = registration_cache.register_local(
        det.image_path,
        other.image_path,
        crop_box,
        expanded_pixel_box(det, scale=1.5),
        expanded_pixel_box(other, scale=1.5),
    )

    cx, cy = det.center_px
    ox, oy = other.center_px
    raw_dx = ox - cx
    raw_dy = oy - cy
    residual_dx = raw_dx - bg_dx
    residual_dy = raw_dy - bg_dy
    local_residual_dx = raw_dx - local_bg_dx
    local_residual_dy = raw_dy - local_bg_dy

    bw, bh = det.size_px
    ow, oh = other.size_px
    avg_w = (bw + ow) / 2
    avg_h = (bh + oh) / 2
    box_diag = max(1.0, math.hypot(avg_w, avg_h))

    return PairMotion(
        frame_gap=float(abs(other.frame_id - det.frame_id)),
        raw_dx=raw_dx,
        raw_dy=raw_dy,
        bg_dx=bg_dx,
        bg_dy=bg_dy,
        local_bg_dx=local_bg_dx,
        local_bg_dy=local_bg_dy,
        residual_dx=residual_dx,
        residual_dy=residual_dy,
        local_residual_dx=local_residual_dx,
        local_residual_dy=local_residual_dy,
        box_diag=box_diag,
        ncc=ncc,
        peak_ratio=peak_ratio,
        local_ncc=local_ncc,
        local_peak_ratio=local_peak_ratio,
    )


def pixel_box(det: Detection) -> tuple[int, int, int, int]:
    """A normalized detection box is converted to full-image pixel corners."""
    cx, cy = det.center_px
    bw, bh = det.size_px
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    return clamp_box((x1, y1, x2, y2), det.image_w, det.image_h)


def expanded_pixel_box(det: Detection, scale: float) -> tuple[int, int, int, int]:
    """The animal mask is expanded slightly so animal heat does not dominate local registration."""
    cx, cy = det.center_px
    bw, bh = det.size_px
    x1 = int(round(cx - bw * scale / 2))
    y1 = int(round(cy - bh * scale / 2))
    x2 = int(round(cx + bw * scale / 2))
    y2 = int(round(cy + bh * scale / 2))
    return clamp_box((x1, y1, x2, y2), det.image_w, det.image_h)


def context_crop_box(det: Detection) -> tuple[int, int, int, int]:
    """
    A local background crop is built around the animal.

    The crop is much larger than the box so nearby background can be used, but
    it is clipped to the image because many difficult examples live near edges.
    """
    cx, cy = det.center_px
    bw, bh = det.size_px
    side = max(96.0, 8.0 * max(bw, bh))
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = int(round(cx + side / 2))
    y2 = int(round(cy + side / 2))
    return clamp_box((x1, y1, x2, y2), det.image_w, det.image_h)


def clamp_box(box: tuple[int, int, int, int], image_w: int, image_h: int) -> tuple[int, int, int, int]:
    """Pixel boxes are clipped and kept at a minimum usable size."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(image_w - 2, x1))
    y1 = max(0, min(image_h - 2, y1))
    x2 = max(x1 + 2, min(image_w, x2))
    y2 = max(y1 + 2, min(image_h, y2))
    return x1, y1, x2, y2


def features_for_detection(
    det: Detection,
    tracks: dict[tuple[str, str, int], list[Detection]],
    registration_cache: ImageRegistrationCache,
    neighbors: int,
) -> np.ndarray:
    """A detection is converted into static image features and temporal features."""
    candidates = neighbor_candidates(det, tracks, neighbors)
    pairs = [pair_motion(det, other, registration_cache) for other in candidates]

    bbox_area = det.w * det.h
    bbox_aspect = det.w / max(det.h, 1e-6)
    edge_distance = min(det.cx, det.cy, 1.0 - det.cx, 1.0 - det.cy)
    has_prev = any(other.frame_id < det.frame_id for other in candidates)
    has_next = any(other.frame_id > det.frame_id for other in candidates)
    base_values = [
        det.cx,
        det.cy,
        det.w,
        det.h,
        bbox_area,
        bbox_aspect,
        float(det.boxes_in_image),
        1.0 if det.track_id is not None else 0.0,
        det.track_iou,
        edge_distance,
        1.0 if edge_distance < max(det.w, det.h) else 0.0,
        1.0 if has_prev else 0.0,
        1.0 if has_next else 0.0,
    ]

    if not pairs:
        temporal_values = [0.0] * (len(FEATURE_NAMES) - len(base_values))
        return np.asarray(base_values + temporal_values, dtype=np.float32)

    frame_gaps = np.asarray([p.frame_gap for p in pairs], dtype=np.float32)
    ncc = np.asarray([p.ncc for p in pairs], dtype=np.float32)
    local_ncc = np.asarray([p.local_ncc for p in pairs], dtype=np.float32)
    peak_log = np.log1p(np.asarray([max(0.0, p.peak_ratio) for p in pairs], dtype=np.float32))
    local_peak_log = np.log1p(np.asarray([max(0.0, p.local_peak_ratio) for p in pairs], dtype=np.float32))
    raw = np.asarray([math.hypot(p.raw_dx, p.raw_dy) / p.box_diag for p in pairs], dtype=np.float32)
    bg = np.asarray([math.hypot(p.bg_dx, p.bg_dy) / p.box_diag for p in pairs], dtype=np.float32)
    local_bg = np.asarray([math.hypot(p.local_bg_dx, p.local_bg_dy) / p.box_diag for p in pairs], dtype=np.float32)
    residual = np.asarray([math.hypot(p.residual_dx, p.residual_dy) / p.box_diag for p in pairs], dtype=np.float32)
    local_residual = np.asarray([math.hypot(p.local_residual_dx, p.local_residual_dy) / p.box_diag for p in pairs], dtype=np.float32)
    bg_disagreement = np.asarray(
        [math.hypot(p.bg_dx - p.local_bg_dx, p.bg_dy - p.local_bg_dy) / p.box_diag for p in pairs],
        dtype=np.float32,
    )
    residual_x = np.asarray([abs(p.residual_dx) / p.box_diag for p in pairs], dtype=np.float32)
    residual_y = np.asarray([abs(p.residual_dy) / p.box_diag for p in pairs], dtype=np.float32)
    local_residual_x = np.asarray([abs(p.local_residual_dx) / p.box_diag for p in pairs], dtype=np.float32)
    local_residual_y = np.asarray([abs(p.local_residual_dy) / p.box_diag for p in pairs], dtype=np.float32)

    # Better-aligned pairs are given more weight without discarding weaker pairs.
    weights = np.clip(np.maximum(ncc, local_ncc), 0.0, None) + 0.05
    weights = weights / max(float(weights.sum()), 1e-6)
    good_global = ncc >= 0.15
    good_local = local_ncc >= 0.15
    prev_count = sum(1 for other in candidates if other.frame_id < det.frame_id)
    next_count = sum(1 for other in candidates if other.frame_id > det.frame_id)
    temporal_quality = float(np.clip((np.maximum(ncc, local_ncc).max() + (1.0 if has_prev and has_next else 0.0)) / 2, 0, 1))

    temporal_values = [
        float(len(pairs)),
        float(prev_count),
        float(next_count),
        float(frame_gaps.mean()),
        float(frame_gaps.min()),
        float(frame_gaps.max()),
        float(ncc.mean()),
        float(ncc.max()),
        float(local_ncc.mean()),
        float(local_ncc.max()),
        float(peak_log.mean()),
        float(peak_log.max()),
        float(local_peak_log.mean()),
        float(local_peak_log.max()),
        float(raw.mean()),
        float(raw.max()),
        float(bg.mean()),
        float(bg.max()),
        float(local_bg.mean()),
        float(local_bg.max()),
        float(residual.mean()),
        float(residual.max()),
        float(local_residual.mean()),
        float(local_residual.max()),
        float(np.sum(weights * residual)),
        float(np.sum(weights * local_residual)),
        float(np.sum(weights * raw)),
        float(np.sum(weights * bg)),
        float(np.sum(weights * local_bg)),
        float(bg_disagreement.mean()),
        float(bg_disagreement.max()),
        float(residual_x.mean()),
        float(residual_y.mean()),
        float(local_residual_x.mean()),
        float(local_residual_y.mean()),
        float(good_global.sum()),
        float(good_local.sum()),
        temporal_quality,
    ]
    return np.asarray(base_values + temporal_values, dtype=np.float32)


def build_feature_matrix(
    detections: list[Detection],
    tracks: dict[tuple[str, str, int], list[Detection]],
    registration_cache: ImageRegistrationCache,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature and target arrays are built for one split."""
    x_rows = []
    y = []
    for index, det in enumerate(detections, start=1):
        x_rows.append(features_for_detection(det, tracks, registration_cache, neighbors))
        y.append(det.motion_id)
        if index % 5000 == 0:
            print(f"  built {index}/{len(detections)} feature rows")
    if not x_rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.vstack(x_rows).astype(np.float32), np.asarray(y, dtype=np.float32)


def load_or_build_feature_matrix(
    split: str,
    detections: list[Detection],
    tracks: dict[tuple[str, str, int], list[Detection]],
    registration_cache: ImageRegistrationCache,
    neighbors: int,
    cache_dir: Path,
    cache_tag: str,
    rebuild_cache: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Feature rows are reused when the cached detection keys match the current run.

    This keeps parameter tuning fast while avoiding stale row/order mismatches.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_{cache_tag}.npz"
    current_keys = np.asarray([detection_cache_key(det) for det in detections])

    if cache_path.exists() and not rebuild_cache:
        cached = np.load(cache_path, allow_pickle=False)
        cached_keys = cached["keys"].astype(str)
        cached_feature_names = cached["feature_names"].astype(str) if "feature_names" in cached else np.asarray([])
        if (
            len(cached_keys) == len(current_keys)
            and np.array_equal(cached_keys, current_keys)
            and np.array_equal(cached_feature_names, np.asarray(FEATURE_NAMES))
        ):
            print(f"  loaded cached {split} features from {cache_path}")
            return cached["x"].astype(np.float32), cached["y"].astype(np.float32)
        print(f"  cache mismatch for {split}; rebuilding features")

    x, y = build_feature_matrix(detections, tracks, registration_cache, neighbors)
    np.savez_compressed(
        cache_path,
        x=x,
        y=y,
        keys=current_keys,
        feature_names=np.asarray(FEATURE_NAMES),
    )
    print(f"  saved {split} features to {cache_path}")
    return x, y
