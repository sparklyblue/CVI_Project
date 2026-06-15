"""PyTorch dataset that builds pairwise thermal crops on demand."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from motion_baseline.common import Detection
from .pairs import PairSample


META_FEATURE_NAMES = [
    "has_neighbor",
    "direction",
    "log_frame_gap",
    "raw_dx_box",
    "raw_dy_box",
    "raw_distance_box",
    "bbox_area",
    "edge_distance_norm",
    "track_match_iou",
]


@dataclass(frozen=True)
class CropSettings:
    """Settings that control how pairwise crops and metadata are built."""

    crop_size: int
    crop_scale: float
    min_crop_pixels: int
    max_crop_pixels: int
    frame_gap_scale: float
    max_motion_box: float


class PairCropDataset(Dataset):
    """Each item returns current/neighbor/difference crop channels and metadata."""

    def __init__(self, samples: list[PairSample], settings: CropSettings):
        self.samples = samples
        self.settings = settings

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        target_crop, neighbor_crop = pair_crops(sample, self.settings)
        diff_crop = np.abs(target_crop - neighbor_crop)

        image = np.stack([target_crop, neighbor_crop, diff_crop], axis=0).astype(np.float32)
        image = (image - 0.5) / 0.5

        meta = pair_metadata(sample, self.settings)
        label = float(sample.target.motion_id)
        return {
            "image": torch.from_numpy(image),
            "meta": torch.from_numpy(meta),
            "label": torch.tensor(label, dtype=torch.float32),
            "target_index": torch.tensor(sample.target_index, dtype=torch.long),
        }


def pair_crops(sample: PairSample, settings: CropSettings) -> tuple[np.ndarray, np.ndarray]:
    """Current and neighbor crops are read from their images using one shared crop box."""
    crop_box = pair_crop_box(sample, settings)
    target_crop = crop_gray(sample.target, crop_box, settings.crop_size)
    if sample.neighbor is None:
        return target_crop, target_crop.copy()
    neighbor_crop = crop_gray(sample.neighbor, crop_box, settings.crop_size)
    return target_crop, neighbor_crop


def pair_crop_box(sample: PairSample, settings: CropSettings) -> tuple[int, int, int, int]:
    """
    A square crop is chosen around the target and its neighbor.

    The same pixel coordinates are used in both images. This preserves the
    apparent image-space displacement, while the crop remains local enough to
    contain nearby background context.
    """
    target_box = pixel_box(sample.target)
    if sample.neighbor is None:
        x1, y1, x2, y2 = target_box
    else:
        neighbor_box = pixel_box(sample.neighbor)
        x1 = min(target_box[0], neighbor_box[0])
        y1 = min(target_box[1], neighbor_box[1])
        x2 = max(target_box[2], neighbor_box[2])
        y2 = max(target_box[3], neighbor_box[3])

    target_w, target_h = sample.target.size_px
    max_box_side = max(target_w, target_h, 1.0)
    union_side = max(x2 - x1, y2 - y1, 1)
    side = max(settings.min_crop_pixels, settings.crop_scale * max_box_side, union_side * 1.25)
    side = min(side, settings.max_crop_pixels)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half = side / 2
    return (
        int(round(cx - half)),
        int(round(cy - half)),
        int(round(cx + half)),
        int(round(cy + half)),
    )


def crop_gray(det: Detection, crop_box: tuple[int, int, int, int], crop_size: int) -> np.ndarray:
    """A grayscale crop is resized and normalized to the 0..1 range."""
    with Image.open(det.image_path) as image:
        image = image.convert("L")
        crop = image.crop(crop_box)
        crop = crop.resize((crop_size, crop_size), Image.Resampling.BILINEAR)
    return np.asarray(crop, dtype=np.float32) / 255.0


def pixel_box(det: Detection) -> tuple[int, int, int, int]:
    """A normalized YOLO box is converted to pixel corner coordinates."""
    cx, cy = det.center_px
    bw, bh = det.size_px
    return (
        int(round(cx - bw / 2)),
        int(round(cy - bh / 2)),
        int(round(cx + bw / 2)),
        int(round(cy + bh / 2)),
    )


def pair_metadata(sample: PairSample, settings: CropSettings) -> np.ndarray:
    """Small numeric pair descriptors are returned alongside the crop channels."""
    det = sample.target
    edge_distance = min(det.cx, det.cy, 1.0 - det.cx, 1.0 - det.cy)
    bbox_area = det.w * det.h

    if sample.neighbor is None:
        values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, bbox_area, edge_distance, det.track_iou]
        return np.asarray(values, dtype=np.float32)

    cx, cy = det.center_px
    ox, oy = sample.neighbor.center_px
    bw, bh = det.size_px
    ow, oh = sample.neighbor.size_px
    box_diag = max(1.0, math.hypot((bw + ow) / 2, (bh + oh) / 2))

    raw_dx = np.clip((ox - cx) / box_diag, -settings.max_motion_box, settings.max_motion_box)
    raw_dy = np.clip((oy - cy) / box_diag, -settings.max_motion_box, settings.max_motion_box)
    raw_dist = np.clip(math.hypot(raw_dx, raw_dy), 0.0, settings.max_motion_box)
    log_gap = math.log1p(sample.frame_gap) / max(math.log1p(settings.frame_gap_scale), 1e-6)

    values = [
        1.0,
        float(sample.direction),
        float(np.clip(log_gap, 0.0, 1.0)),
        float(raw_dx / settings.max_motion_box),
        float(raw_dy / settings.max_motion_box),
        float(raw_dist / settings.max_motion_box),
        bbox_area,
        edge_distance,
        det.track_iou,
    ]
    return np.asarray(values, dtype=np.float32)

