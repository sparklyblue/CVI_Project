"""Visual error panels for false-positive and false-negative inspection."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import MOVING, STATIC, Detection


def export_error_panels(
    split: str,
    detections: list[Detection],
    probs: np.ndarray,
    threshold: float,
    tracks: dict[tuple[str, str, int], list[Detection]],
    output_dir: Path,
    max_errors: int,
) -> None:
    """The highest-confidence false positives/negatives are saved as image panels."""
    if max_errors <= 0:
        return

    split_dir = output_dir / "error_panels" / split
    fp_dir = split_dir / "false_positive"
    fn_dir = split_dir / "false_negative"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    false_positives = []
    false_negatives = []
    for index, (det, prob) in enumerate(zip(detections, probs)):
        pred = MOVING if prob >= threshold else STATIC
        if pred == MOVING and det.motion_id == STATIC:
            false_positives.append((float(prob), index, det))
        elif pred == STATIC and det.motion_id == MOVING:
            false_negatives.append((1.0 - float(prob), index, det))

    false_positives.sort(reverse=True, key=lambda item: item[0])
    false_negatives.sort(reverse=True, key=lambda item: item[0])

    rows = []
    rows += write_error_group("false_positive", false_positives[:max_errors], probs, threshold, tracks, fp_dir)
    rows += write_error_group("false_negative", false_negatives[:max_errors], probs, threshold, tracks, fn_dir)

    with (split_dir / "error_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "error_type",
                "file",
                "stem",
                "flight_id",
                "frame_id",
                "row_id",
                "track_id",
                "true_motion",
                "p_moving",
                "threshold",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_error_group(
    error_type: str,
    errors: list[tuple[float, int, Detection]],
    probs: np.ndarray,
    threshold: float,
    tracks: dict[tuple[str, str, int], list[Detection]],
    output_dir: Path,
) -> list[dict[str, str | int | float]]:
    """One false-positive or false-negative group is written to disk."""
    rows = []
    for rank, (_severity, index, det) in enumerate(errors, start=1):
        filename = f"{rank:03d}_{det.stem}_row{det.row_id}.jpg"
        panel = make_error_panel(det, float(probs[index]), threshold, error_type, tracks)
        panel.save(output_dir / filename, quality=90)
        rows.append(
            {
                "error_type": error_type,
                "file": str(output_dir / filename),
                "stem": det.stem,
                "flight_id": det.flight_id,
                "frame_id": det.frame_id,
                "row_id": det.row_id,
                "track_id": "" if det.track_id is None else det.track_id,
                "true_motion": det.motion_id,
                "p_moving": f"{float(probs[index]):.6f}",
                "threshold": f"{threshold:.6f}",
            }
        )
    return rows


def make_error_panel(
    det: Detection,
    prob: float,
    threshold: float,
    error_type: str,
    tracks: dict[tuple[str, str, int], list[Detection]],
) -> Image.Image:
    """A previous/current/next panel is assembled for one mistaken prediction."""
    previous_det, next_det = track_neighbors(det, tracks)
    panels = [
        draw_detection_image(previous_det, "previous") if previous_det else blank_panel("previous"),
        draw_detection_image(det, "current", highlight=True),
        draw_detection_image(next_det, "next") if next_det else blank_panel("next"),
    ]

    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels) + 70
    canvas = Image.new("RGB", (width, height), color=(20, 20, 20))
    x_offset = 0
    for panel in panels:
        canvas.paste(panel, (x_offset, 0))
        x_offset += panel.width

    draw = ImageDraw.Draw(canvas)
    true_name = "moving" if det.motion_id == MOVING else "static"
    pred_name = "moving" if prob >= threshold else "static"
    text = (
        f"{error_type} | {det.stem} row {det.row_id} | "
        f"true={true_name} pred={pred_name} p_moving={prob:.3f} threshold={threshold:.2f}"
    )
    draw.rectangle([0, height - 70, width, height], fill=(20, 20, 20))
    draw.text((10, height - 52), text, fill=(255, 255, 255))
    draw.text((10, height - 28), f"track_id={det.track_id}  box=(cx={det.cx:.3f}, cy={det.cy:.3f}, w={det.w:.3f}, h={det.h:.3f})", fill=(210, 210, 210))
    return canvas


def track_neighbors(
    det: Detection,
    tracks: dict[tuple[str, str, int], list[Detection]],
) -> tuple[Detection | None, Detection | None]:
    """Immediate previous and next detections are found from the recovered track."""
    if det.track_id is None:
        return None, None
    track = tracks.get((det.split, det.flight_id, det.track_id), [])
    for index, candidate in enumerate(track):
        if candidate.stem == det.stem and candidate.row_id == det.row_id:
            previous_det = track[index - 1] if index > 0 else None
            next_det = track[index + 1] if index + 1 < len(track) else None
            return previous_det, next_det
    return None, None


def draw_detection_image(det: Detection, label: str, highlight: bool = False) -> Image.Image:
    """A detection box is drawn on a resized image."""
    with Image.open(det.image_path) as img:
        img = img.convert("RGB")
        original_w, original_h = img.size
        scale = min(1.0, 420 / max(original_w, original_h))
        new_size = (int(original_w * scale), int(original_h * scale))
        if new_size != img.size:
            img = img.resize(new_size, Image.Resampling.BILINEAR)

    draw = ImageDraw.Draw(img)
    x1 = (det.cx - det.w / 2) * img.width
    y1 = (det.cy - det.h / 2) * img.height
    x2 = (det.cx + det.w / 2) * img.width
    y2 = (det.cy + det.h / 2) * img.height
    color = (255, 220, 40) if highlight else (70, 210, 255)
    line_width = 4 if highlight else 2
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((6, 4), f"{label}: {det.stem}", fill=(255, 255, 255))
    return img


def blank_panel(label: str) -> Image.Image:
    """A placeholder is used when a track neighbor is unavailable."""
    img = Image.new("RGB", (420, 315), color=(40, 40, 40))
    draw = ImageDraw.Draw(img)
    draw.text((12, 14), f"{label}: no track neighbor", fill=(230, 230, 230))
    return img
