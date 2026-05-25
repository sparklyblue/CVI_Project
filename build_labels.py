"""
BAMBI Auto-Labeller
====================
Joins mots ground-truth files with existing YOLO label files to produce
properly labelled training data with real species classes and motion state.

Output per image (same YOLO format, extended):
    species_id  cx  cy  w  h  motion_id

Species IDs  (also written to data.yaml):
    0  Capreolus capreolus (Roe deer)
    1  Cervus elaphus (Red deer)
    2  Dama dama (Fallow Deer)
    3  Sus scrofa (Wild boar)
    4  Capra ibex (Alpine ibex)
    5  Rupicapra rupicapra (Chamois)
    6  Sus scrofa x Sus domesticus (Hybrid Pig)
    7  Homo sapiens (Human)
    8  Aves (Bird)
    9  Canis lupus familiaris (Dog)
    10 Unknown
    11 No-animal

Motion IDs:
    0  static
    1  moving
    2  ambiguous

Usage:
    py build_labels.py
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

IMAGES_DIR          = Path("images_thermal/images")       # contains train/val/test
YOLO_LABELS_DIR     = Path("labels_matched_thermal")      # contains train/val/test
MOTS_DIR            = Path("mots")                        # flat folder of N_gt.txt files
OUTPUT_LABELS_DIR   = Path("labels_final")                # output (same train/val/test structure)
SUMMARY_CSV         = Path("label_summary.csv")
DATA_YAML           = Path("data.yaml")

SPLITS = ["train", "val", "test"]

# IoU threshold for matching a YOLO box to a mots detection
IOU_THRESHOLD = 0.05  

# ── SPECIES MAP ───────────────────────────────────────────────────────────────

# Maps the species string in col 10 of mots to a class ID
SPECIES_MAP = {
    "Capreolus capreolus (Roe deer)":                   0,
    "Cervus elaphus (Red deer)":                        1,
    "Dama dama (Fallow Deer)":                          2,
    "Sus scrofa (Wild boar)":                           3,
    "Capra ibex (Alpine ibex)":                         4,
    "Rupicapra rupicapra (Chamois)":                    5,
    "Sus scrofa x Sus domesticus (Hybrid Pig)":         6,
    "Homo sapiens (Human)":                             7,
    "Aves (Bird)":                                      8,
    "Canis lupus familiaris (Dog)":                     9,
    "Unknown":                                          10,
    "No-animal":                                        11,
}

# The shorter names for output
SPECIES_NAMES = [
    "Roe deer",
    "Red deer",
    "Fallow Deer",
    "Wild boar",
    "Alpine ibex",
    "Chamois",
    "Hybrid Pig",
    "Human",
    "Bird",
    "Dog",
    "Unknown",
    "No-animal",
]

# Motion: col values
MOTION_NAMES = {0: "static", 1: "moving", 2: "ambiguous"}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_mots(mots_dir: Path) -> dict:
    """
    Returns a nested dict:
        mots[flight_id][frame_id] = list of dicts with keys:
            track_id, x, y, w, h, species_id, motion_id, species_name
    Pixel coords (not normalised).
    """
    mots = {}
    files = list(mots_dir.glob("*_gt.txt"))
    print(f"Loading {len(files)} mots files...")
    for f in files:
        flight_id = f.stem.replace("_gt", "")
        frames = defaultdict(list)
        with open(f, newline="", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                cols = line.split(",")
                if len(cols) < 12:
                    continue
                try:
                    frame_id  = int(cols[0])
                    track_id  = int(cols[1])
                    x         = float(cols[2])
                    y         = float(cols[3])
                    w         = float(cols[4])
                    h         = float(cols[5])
                    species_name = cols[9].strip()
                    motion_id = int(cols[10])   # col 11 (0-indexed col 10)
                except (ValueError, IndexError):
                    continue

                species_id = SPECIES_MAP.get(species_name, 10)  # default Unknown
                frames[frame_id].append({
                    "track_id":    track_id,
                    "x": x, "y": y, "w": w, "h": h,
                    "species_id":  species_id,
                    "species_name": species_name,
                    "motion_id":   motion_id,
                })
        mots[flight_id] = dict(frames)
    return mots

def iou(box_a, box_b):
    """
    box_a: (cx, cy, w, h) normalised  [YOLO format]
    box_b: (x, y, w, h)  pixel coords [mots format] — will be normalised before call
    Both are normalised here.
    """
    def to_corners(cx, cy, w, h):
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    ax1, ay1, ax2, ay2 = to_corners(*box_a)
    bx1, by1, bx2, by2 = to_corners(*box_b)

    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def get_image_size(image_path: Path):
    """Returns (width, height) without loading the full image."""
    # Try fast header-only read with struct for JPEG/PNG
    try:
        import struct
        with open(image_path, "rb") as f:
            header = f.read(26)
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", header[16:24])
            return w, h
        if header[:2] == b"\xff\xd8":  # JPEG — need to scan
            with open(image_path, "rb") as f:
                f.read(2)
                while True:
                    marker, = struct.unpack(">H", f.read(2))
                    length, = struct.unpack(">H", f.read(2))
                    if marker in (0xFFC0, 0xFFC2):
                        f.read(1)
                        h, w = struct.unpack(">HH", f.read(4))
                        return w, h
                    f.read(length - 2)
    except Exception:
        pass
    # Fallback: PIL
    try:
        from PIL import Image as PILImage
        with PILImage.open(image_path) as im:
            return im.size  # (w, h)
    except Exception:
        return None, None


def match_boxes(yolo_boxes, mots_detections, img_w, img_h):
    """
    For each YOLO box (cx, cy, w, h normalised), find the best-matching
    mots detection by IoU and return its species_id and motion_id.
    Returns list of (species_id, motion_id) parallel to yolo_boxes.
    Unmatched boxes get (10, 2) = Unknown / ambiguous.
    """
    # Normalise mots boxes
    norm_mots = []
    for det in mots_detections:
        if img_w and img_h:
            cx = (det["x"] + det["w"] / 2) / img_w
            cy = (det["y"] + det["h"] / 2) / img_h
            nw = det["w"] / img_w
            nh = det["h"] / img_h
        else:
            cx = cy = nw = nh = 0
        norm_mots.append((cx, cy, nw, nh, det["species_id"], det["motion_id"]))

    used = set()
    results = []
    for ybox in yolo_boxes:
        best_iou = IOU_THRESHOLD
        best_idx = -1
        for i, mbox in enumerate(norm_mots):
            if i in used:
                continue
            score = iou(ybox, mbox[:4])
            if score > best_iou:
                best_iou = score
                best_idx = i
        if best_idx >= 0:
            used.add(best_idx)
            results.append((norm_mots[best_idx][4], norm_mots[best_idx][5]))
        else:
            results.append((10, 2))  # Unknown, ambiguous
    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    mots = load_mots(MOTS_DIR)

    summary_rows = []
    stats = {
        "total_images": 0,
        "total_boxes": 0,
        "matched": 0,
        "species_counts": defaultdict(int),
        "motion_counts": defaultdict(int),
        "no_mots_file": 0,
        "no_mots_frame": 0,
    }

    for split in SPLITS:
        img_dir   = IMAGES_DIR / split
        label_dir = YOLO_LABELS_DIR / split
        out_dir   = OUTPUT_LABELS_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        # Drive from images — every image gets a label file
        image_files = sorted(
            f for f in img_dir.glob("*")
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        print(f"\n{split}: {len(image_files)} images")

        from_yolo = 0
        from_mots = 0
        skipped   = 0

        for img_path in image_files:
            stem  = img_path.stem   # e.g. "159_2187"
            parts = stem.split("_", 1)
            if len(parts) != 2:
                skipped += 1
                continue

            flight_id, frame_id_str = parts[0], parts[1]
            try:
                frame_id = int(frame_id_str)
            except ValueError:
                skipped += 1
                continue

            img_w, img_h = get_image_size(img_path)

            # Get mots detections for this frame (needed in both paths)
            mots_dets = []
            if flight_id not in mots:
                stats["no_mots_file"] += 1
            elif frame_id not in mots[flight_id]:
                stats["no_mots_frame"] += 1
            else:
                mots_dets = mots[flight_id][frame_id]

            out_path = out_dir / (stem + ".txt")

            # ── PATH A: label file exists → IoU-match to get species/motion ──
            yolo_label = label_dir / (stem + ".txt")
            if yolo_label.exists():
                yolo_boxes = []
                with open(yolo_label, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        vals = line.split()
                        if len(vals) < 5:
                            continue
                        cx, cy, w, h = float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])
                        yolo_boxes.append((cx, cy, w, h))

                if mots_dets and yolo_boxes:
                    matches = match_boxes(yolo_boxes, mots_dets, img_w, img_h)
                else:
                    matches = [(10, 2)] * len(yolo_boxes)

                with open(out_path, "w", encoding="utf-8") as out:
                    for (cx, cy, w, h), (sp_id, mo_id) in zip(yolo_boxes, matches):
                        out.write(f"{sp_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {mo_id}\n")
                        stats["total_boxes"] += 1
                        stats["species_counts"][sp_id] += 1
                        stats["motion_counts"][mo_id] += 1
                        stats["matched"] += 1

                from_yolo += 1
                box_count = len(yolo_boxes)

            # ── PATH B: no label file → generate directly from mots ──────────
            else:
                if not mots_dets:
                    # No label file and no mots entry → empty label (background frame)
                    out_path.write_text("")
                    stats["total_images"] += 1
                    summary_rows.append({
                        "split": split, "image": stem, "flight": flight_id,
                        "frame": frame_id, "boxes": 0, "source": "empty",
                    })
                    continue

                if img_w is None or img_h is None:
                    skipped += 1
                    continue

                with open(out_path, "w", encoding="utf-8") as out:
                    for det in mots_dets:
                        cx = (det["x"] + det["w"] / 2) / img_w
                        cy = (det["y"] + det["h"] / 2) / img_h
                        nw = det["w"] / img_w
                        nh = det["h"] / img_h
                        sp_id = det["species_id"]
                        mo_id = det["motion_id"]
                        out.write(f"{sp_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} {mo_id}\n")
                        stats["total_boxes"] += 1
                        stats["species_counts"][sp_id] += 1
                        stats["motion_counts"][mo_id] += 1
                        stats["matched"] += 1

                from_mots += 1
                box_count = len(mots_dets)

            stats["total_images"] += 1
            summary_rows.append({
                "split":   split,
                "image":   stem,
                "flight":  flight_id,
                "frame":   frame_id,
                "boxes":   box_count,
                "source":  "yolo+mots" if yolo_label.exists() else "mots_only",
            })

        print(f"  from yolo+mots : {from_yolo}")
        print(f"  from mots only : {from_mots}")
        print(f"  skipped        : {skipped}")

    # ── Write summary CSV ──────────────────────────────────────────────────
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split","image","flight","frame","boxes","source"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary CSV written to {SUMMARY_CSV}")

    # ── Write data.yaml ───────────────────────────────────────────────────
    yaml_lines = [
        f"path: .",
        f"train: images_thermal/images/train",
        f"val:   images_thermal/images/val",
        f"test:  images_thermal/images/test",
        f"",
        f"# Label format per box: species_id cx cy w h motion_id",
        f"# motion: 0=static  1=moving  2=ambiguous",
        f"",
        f"nc: {len(SPECIES_NAMES)}",
        f"names:",
    ]
    for name in SPECIES_NAMES:
        yaml_lines.append(f"  - {name}")
    yaml_lines += [
        "",
        "motion_nc: 3",
        "motion_names:",
        "  - static",
        "  - moving",
        "  - ambiguous",
    ]
    DATA_YAML.write_text("\n".join(yaml_lines), encoding="utf-8")
    print(f"data.yaml written to {DATA_YAML}")

    # ── Print stats ───────────────────────────────────────────────────────
    print("\n" + "="*52)
    print("DONE")
    print("="*52)
    print(f"  Images processed : {stats['total_images']}")
    print(f"  Boxes total      : {stats['total_boxes']}")
    print(f"  Matched to mots  : {stats['matched']}")
    print(f"  No mots file     : {stats['no_mots_file']}")
    print(f"  No mots frame    : {stats['no_mots_frame']}")
    print("\nSpecies distribution:")
    for sp_id, count in sorted(stats["species_counts"].items()):
        print(f"  {sp_id:2d}  {SPECIES_NAMES[sp_id]:<30s} {count:6d}")
    print("\nMotion distribution:")
    for mo_id, count in sorted(stats["motion_counts"].items()):
        print(f"  {mo_id}  {MOTION_NAMES[mo_id]:<12s} {count:6d}")
    print(f"\nOutput labels : {OUTPUT_LABELS_DIR}/")
    print(f"Summary CSV   : {SUMMARY_CSV}")
    print(f"data.yaml     : {DATA_YAML}")


if __name__ == "__main__":
    main()