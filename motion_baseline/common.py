"""Shared constants, data structures, and small parsing helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


SPLITS = ["train", "val", "test"]

IMAGES_DIR = Path("images_thermal/images")
LABELS_DIR = Path("labels_filtered")
MOTS_DIR = Path("mots")
DEFAULT_OUTPUT_DIR = Path("dist/motion_baseline")

STATIC = 0
MOVING = 1

SPECIES_MAP = {
    "Capreolus capreolus (Roe deer)": 0,
    "Cervus elaphus (Red deer)": 1,
    "Dama dama (Fallow Deer)": 2,
    "Sus scrofa (Wild boar)": 3,
    "Capra ibex (Alpine ibex)": 4,
    "Rupicapra rupicapra (Chamois)": 5,
    "Sus scrofa x Sus domesticus (Hybrid Pig)": 6,
    "Homo sapiens (Human)": 7,
    "Aves (Bird)": 8,
    "Canis lupus familiaris (Dog)": 9,
    "Unknown": 10,
    "No-animal": 11,
}

# The filtered labels remap the hybrid pig class from original id 6 to id 4.
FILTERED_SPECIES_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 6: 4}


@dataclass(frozen=True)
class MotDetection:
    """
    A raw MOTS detection is stored in pixel coordinates.

    These detections are not used directly for training labels. They are used
    to recover track ids that were lost in the filtered YOLO label files.
    """

    flight_id: str
    frame_id: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    filtered_species_id: int | None
    motion_id: int


@dataclass
class Detection:
    """
    A filtered YOLO detection is stored in normalized coordinates.

    This is the central row object used by the baseline: one object represents
    one animal box in one image, plus the static/moving target label.
    """

    split: str
    stem: str
    flight_id: str
    frame_id: int
    row_id: int
    species_id: int
    cx: float
    cy: float
    w: float
    h: float
    motion_id: int
    image_path: Path
    label_path: Path
    image_w: int
    image_h: int
    boxes_in_image: int
    track_id: int | None = None
    track_iou: float = 0.0

    @property
    def center_px(self) -> tuple[float, float]:
        return self.cx * self.image_w, self.cy * self.image_h

    @property
    def size_px(self) -> tuple[float, float]:
        return self.w * self.image_w, self.h * self.image_h


def detection_cache_key(det: Detection) -> str:
    """
    A stable row key is used to validate cached feature matrices.

    Cached features are only reused when the same detections appear in the same
    order. This prevents old feature rows from silently attaching to new labels.
    """
    track = "" if det.track_id is None else str(det.track_id)
    return "|".join(
        [
            det.split,
            det.stem,
            str(det.row_id),
            str(det.motion_id),
            track,
        ]
    )


def parse_csv_list(raw: str, convert: Callable[[str], object], value_name: str) -> list[object]:
    """A comma-separated CLI value is split and converted into typed values."""
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(convert(item))
    if not values:
        raise ValueError(f"At least one {value_name} value must be provided.")
    return values


def parse_float_list(raw: str) -> list[float]:
    """A comma-separated CLI value is parsed into floats."""
    return [float(value) for value in parse_csv_list(raw, float, "float")]


def parse_int_list(raw: str) -> list[int]:
    """A comma-separated CLI value is parsed into integers."""
    return [int(value) for value in parse_csv_list(raw, int, "integer")]


def parse_optional_int_list(raw: str) -> list[int | None]:
    """A comma-separated CLI value is parsed into integers or None."""
    return [
        None if str(value).lower() == "none" else int(value)
        for value in parse_csv_list(raw, str, "integer/None")
    ]


def parse_optional_str_list(raw: str) -> list[str | None]:
    """A comma-separated CLI value is parsed into strings or None."""
    return [
        None if str(value).lower() == "none" else str(value)
        for value in parse_csv_list(raw, str, "string/None")
    ]
