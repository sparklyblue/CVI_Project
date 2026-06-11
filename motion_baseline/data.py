"""Dataset loading and MOTS track recovery."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .common import (
    FILTERED_SPECIES_REMAP,
    MOVING,
    SPECIES_MAP,
    SPLITS,
    STATIC,
    Detection,
    MotDetection,
)


def load_mots(mots_dir: Path) -> dict[str, dict[int, list[MotDetection]]]:
    """Raw MOTS files are loaded by flight and frame for later track recovery."""
    mots: dict[str, dict[int, list[MotDetection]]] = {}
    for path in sorted(mots_dir.glob("*_gt.txt")):
        flight_id = path.stem.replace("_gt", "")
        frames: dict[int, list[MotDetection]] = defaultdict(list)
        with path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                cols = line.split(",")
                if len(cols) < 11:
                    continue
                try:
                    frame_id = int(cols[0])
                    track_id = int(cols[1])
                    x = float(cols[2])
                    y = float(cols[3])
                    w = float(cols[4])
                    h = float(cols[5])
                    species_name = cols[9].strip()
                    motion_id = int(cols[10])
                except ValueError:
                    continue

                original_species_id = SPECIES_MAP.get(species_name)
                filtered_species_id = FILTERED_SPECIES_REMAP.get(original_species_id)
                frames[frame_id].append(
                    MotDetection(
                        flight_id=flight_id,
                        frame_id=frame_id,
                        track_id=track_id,
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        filtered_species_id=filtered_species_id,
                        motion_id=motion_id,
                    )
                )
        mots[flight_id] = dict(frames)
    return mots


def image_size(path: Path) -> tuple[int, int]:
    """The image size is read once for normalized-to-pixel conversions."""
    with Image.open(path) as img:
        return img.size


def find_image(images_dir: Path, split: str, stem: str) -> Path | None:
    """The matching image path is found for common image extensions."""
    split_dir = images_dir / split
    for suffix in [".jpg", ".jpeg", ".png"]:
        candidate = split_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_filtered_detections(images_dir: Path, labels_dir: Path) -> dict[str, list[Detection]]:
    """Filtered YOLO labels are loaded as one training example per animal box."""
    by_split: dict[str, list[Detection]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        label_files = sorted((labels_dir / split).glob("*.txt"))
        for label_path in label_files:
            lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
            if not lines:
                continue

            image_path = find_image(images_dir, split, label_path.stem)
            if image_path is None:
                continue

            parts = label_path.stem.split("_", 1)
            if len(parts) != 2:
                continue
            flight_id, frame_text = parts
            try:
                frame_id = int(frame_text)
            except ValueError:
                continue

            img_w, img_h = image_size(image_path)
            boxes_in_image = len(lines)
            for row_id, line in enumerate(lines):
                values = line.split()
                if len(values) < 6:
                    continue
                motion_id = int(values[5])
                if motion_id not in (STATIC, MOVING):
                    continue
                by_split[split].append(
                    Detection(
                        split=split,
                        stem=label_path.stem,
                        flight_id=flight_id,
                        frame_id=frame_id,
                        row_id=row_id,
                        species_id=int(values[0]),
                        cx=float(values[1]),
                        cy=float(values[2]),
                        w=float(values[3]),
                        h=float(values[4]),
                        motion_id=motion_id,
                        image_path=image_path,
                        label_path=label_path,
                        image_w=img_w,
                        image_h=img_h,
                        boxes_in_image=boxes_in_image,
                    )
                )
    return by_split


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """IoU is computed for normalized YOLO boxes."""
    ax1, ay1, ax2, ay2 = yolo_to_corners(*box_a)
    bx1, by1, bx2, by2 = yolo_to_corners(*box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def yolo_to_corners(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    """A normalized YOLO box is converted to normalized corner coordinates."""
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def mot_to_yolo(det: MotDetection, image_w: int, image_h: int) -> tuple[float, float, float, float]:
    """A MOTS pixel box is converted to normalized YOLO coordinates."""
    cx = (det.x + det.w / 2) / image_w
    cy = (det.y + det.h / 2) / image_h
    w = det.w / image_w
    h = det.h / image_h
    return cx, cy, w, h


def attach_track_ids(
    detections_by_split: dict[str, list[Detection]],
    mots: dict[str, dict[int, list[MotDetection]]],
) -> None:
    """
    Track ids are recovered by matching filtered boxes back to MOTS boxes.

    Species is checked here only to avoid matching the wrong MOTS object when
    several animals are present. It is not added to the model feature vector.
    """
    for detections in detections_by_split.values():
        by_frame: dict[tuple[str, int, str], list[Detection]] = defaultdict(list)
        for det in detections:
            by_frame[(det.flight_id, det.frame_id, det.split)].append(det)

        for (flight_id, frame_id, _split), frame_dets in by_frame.items():
            mot_dets = mots.get(flight_id, {}).get(frame_id, [])
            used_mots: set[int] = set()
            for det in frame_dets:
                best_index = None
                best_score = 0.0
                label_box = (det.cx, det.cy, det.w, det.h)
                for idx, mot_det in enumerate(mot_dets):
                    if idx in used_mots:
                        continue
                    if mot_det.filtered_species_id != det.species_id:
                        continue
                    if mot_det.motion_id != det.motion_id:
                        continue
                    mot_box = mot_to_yolo(mot_det, det.image_w, det.image_h)
                    score = bbox_iou(label_box, mot_box)
                    if score > best_score:
                        best_score = score
                        best_index = idx
                if best_index is not None:
                    used_mots.add(best_index)
                    det.track_id = mot_dets[best_index].track_id
                    det.track_iou = best_score


def build_track_index(detections_by_split: dict[str, list[Detection]]) -> dict[tuple[str, str, int], list[Detection]]:
    """
    Detections are grouped by split, flight, and track.

    Splits are kept separate so training features do not use validation or test
    images as temporal context.
    """
    tracks: dict[tuple[str, str, int], list[Detection]] = defaultdict(list)
    for split, detections in detections_by_split.items():
        for det in detections:
            if det.track_id is None:
                continue
            tracks[(split, det.flight_id, det.track_id)].append(det)

    for track_dets in tracks.values():
        track_dets.sort(key=lambda item: (item.frame_id, item.stem, item.row_id))
    return dict(tracks)


def stratified_limit(detections: list[Detection], limit: int, seed: int) -> list[Detection]:
    """A balanced-ish debug subset is selected when a max split size is requested."""
    if limit <= 0 or len(detections) <= limit:
        return detections

    rng = random.Random(seed)
    by_class: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        by_class[det.motion_id].append(det)
    for items in by_class.values():
        rng.shuffle(items)

    classes = sorted(by_class)
    per_class = max(1, limit // len(classes))
    chosen = []
    for cls in classes:
        chosen.extend(by_class[cls][:per_class])

    remaining = [item for cls in classes for item in by_class[cls][per_class:]]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, limit - len(chosen))])
    rng.shuffle(chosen)
    return chosen


def summarize_split(name: str, detections: list[Detection]) -> None:
    """Split class counts are printed before training."""
    counts = Counter(det.motion_id for det in detections)
    tracked = sum(1 for det in detections if det.track_id is not None)
    print(
        f"{name}: detections={len(detections)} "
        f"static={counts.get(STATIC, 0)} moving={counts.get(MOVING, 0)} "
        f"track_recovered={tracked}"
    )
