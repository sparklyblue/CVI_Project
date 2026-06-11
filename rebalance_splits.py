"""
BAMBI Split Rebalancer
=======================
Moves whole flights between train/val/test splits to ensure all 5 species
are represented in val and test.
 
Moves images and labels_filtered files directly — no remapping needed
since labels_filtered already has clean IDs (0-4).
 
Flights being moved TO val:
    142  — Wild boar + Hybrid Pig  (289 images)
    143  — Red deer + Fallow Deer  (287 images)
    246  — Wild boar + Hybrid Pig  (~724 images)
 
Flights being moved TO test:
    140  — Wild boar + Hybrid Pig  (329 images)
    135  — Wild boar + Hybrid Pig
    252  — Red deer + Fallow Deer  (small)
 
Usage:
    python rebalance_splits.py
 
    Run build_labels.py and filter_labels.py first if not already done.
    This script moves files in-place — back up your data first if unsure.
"""

import shutil
from pathlib import Path
 
# ── CONFIG ────────────────────────────────────────────────────────────────────
 
IMAGES_DIR          = Path("images_thermal/images")
LABELS_FILTERED_DIR = Path("labels_filtered")
 
# Flights to move: (flight_id, from_split, to_split)
# Looked at all the flights + labels - what's needed where and to increase the % of test & val again
MOVES = [
    # → val: need Wild boar, Hybrid Pig, Fallow Deer
    ("142", "train", "val"),
    ("143", "train", "val"),
    ("246", "train", "val"),
    # → test: need Wild boar, Hybrid Pig, Fallow Deer
    ("140", "train", "test"),
    ("135", "train", "test"),
    ("252", "train", "test"),
    ("250", "train", "test"),
]
 
# ── HELPERS ───────────────────────────────────────────────────────────────────
 
def move_flight_files(flight_id, from_split, to_split, base_dir):
    """Move all files for a flight from one split folder to another."""
    src = base_dir / from_split
    dst = base_dir / to_split
    dst.mkdir(parents=True, exist_ok=True)
 
    files = list(src.glob(f"{flight_id}_*"))
    for f in files:
        shutil.move(str(f), dst / f.name)
    return len(files)
 
 
# ── MAIN ──────────────────────────────────────────────────────────────────────
 
def main():
    print("BAMBI Split Rebalancer")
    print("=" * 52)
 
    for flight_id, from_split, to_split in MOVES:
        print(f"\nFlight {flight_id}: {from_split} → {to_split}")
 
        n_images = move_flight_files(flight_id, from_split, to_split, IMAGES_DIR)
        print(f"  Images moved          : {n_images}")
 
        n_labels = move_flight_files(flight_id, from_split, to_split, LABELS_FILTERED_DIR)
        print(f"  labels_filtered moved : {n_labels}")
 
    # ── Print final distribution ──────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("DONE")
    print("=" * 52)
 
    print("\nNew split sizes (images):")
    for split in ["train", "val", "test"]:
        n = len(list((IMAGES_DIR / split).glob("*.jpg")))
        print(f"  {split:<6} : {n}")
 
    print("\nNew species distribution per split (labels_filtered):")
    names = {0: "Roe deer", 1: "Red deer", 2: "Fallow Deer",
             3: "Wild boar", 4: "Hybrid Pig"}
    for split in ["train", "val", "test"]:
        counts = {}
        for lf in (LABELS_FILTERED_DIR / split).glob("*.txt"):
            for line in lf.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    sp = int(parts[0])
                    counts[sp] = counts.get(sp, 0) + 1
        summary = ", ".join(f"{names[k]}:{v}" for k, v in sorted(counts.items()))
        print(f"  {split:<6} : {summary if summary else 'empty'}")
 
 
if __name__ == "__main__":
    main()
