"""Shared constants and data structures for the motion baseline."""

from __future__ import annotations

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
    """A raw MOTS detection is stored in pixel coordinates."""

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
    """A filtered YOLO detection is stored in normalized coordinates."""

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
    """A stable row key is used to validate cached feature matrices."""
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


def parse_float_list(raw: str) -> list[float]:
    """A comma-separated CLI value is parsed into floats."""
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("At least one float value must be provided.")
    return values
