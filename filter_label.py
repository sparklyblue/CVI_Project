"""
BAMBI Dataset Filter
=====================
Filters labels_final/ to produce a clean subset for training with:
  - Only 5 species (remapped to IDs 0-4)
  - Only static (0) and moving (1) motion — no ambiguous (2)
  - Images are dropped entirely if ANY box is ambiguous or an excluded species

Output: labels_filtered/ and data_filtered.yaml

Species mapping:
    Old ID → New ID   Name
       0   →   0      Roe deer
       1   →   1      Red deer
       2   →   2      Fallow Deer
       3   →   3      Wild boar
       6   →   4      Hybrid Pig

Usage:
    python filter_labels.py
"""

import csv
import shutil
from collections import defaultdict
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_LABELS_DIR  = Path("labels_final")
OUTPUT_LABELS_DIR = Path("labels_filtered")
DATA_YAML         = Path("data_filtered.yaml")
SUMMARY_CSV       = Path("filter_summary.csv")

SPLITS = ["train", "val", "test"]

# Old species ID → new species ID (anything not listed is excluded)
SPECIES_REMAP = {
    0: 0,   # Roe deer
    1: 1,   # Red deer
    2: 2,   # Fallow Deer
    3: 3,   # Wild boar
    6: 4,   # Hybrid Pig
}

SPECIES_NAMES = [
    "Roe deer",
    "Red deer",
    "Fallow Deer",
    "Wild boar",
    "Hybrid Pig",
]

MOTION_NAMES = {0: "static", 1: "moving"}

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    stats = {
        "total_in":          0,
        "kept":              0,
        "dropped_ambiguous": 0,
        "dropped_species":   0,
        "boxes_in":          0,
        "boxes_out":         0,
        "species_counts":    defaultdict(int),
        "motion_counts":     defaultdict(int),
    }

    summary_rows = []

    for split in SPLITS:
        in_dir  = INPUT_LABELS_DIR / split
        out_dir = OUTPUT_LABELS_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        label_files = sorted(in_dir.glob("*.txt"))
        kept = dropped_amb = dropped_sp = 0

        for lf in label_files:
            stats["total_in"] += 1

            lines = [l.strip() for l in lf.read_text().splitlines() if l.strip()]

            # Empty file = background frame, always keep
            if not lines:
                (out_dir / lf.name).write_text("")
                kept += 1
                stats["kept"] += 1
                summary_rows.append({
                    "split": split, "image": lf.stem,
                    "kept": True, "reason": "background", "boxes_out": 0,
                })
                continue

            # Parse all boxes
            boxes = []
            for line in lines:
                parts = line.split()
                if len(parts) < 6:
                    continue
                sp_id = int(parts[0])
                cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]
                mo_id = int(parts[5])
                boxes.append((sp_id, cx, cy, w, h, mo_id))

            stats["boxes_in"] += len(boxes)

            # Drop whole image if ANY box is ambiguous motion
            if any(mo == 2 for _, _, _, _, _, mo in boxes):
                dropped_amb += 1
                stats["dropped_ambiguous"] += 1
                summary_rows.append({
                    "split": split, "image": lf.stem,
                    "kept": False, "reason": "ambiguous_motion", "boxes_out": 0,
                })
                continue

            # Drop whole image if ANY box is an excluded species
            if any(sp not in SPECIES_REMAP for sp, *_ in boxes):
                dropped_sp += 1
                stats["dropped_species"] += 1
                summary_rows.append({
                    "split": split, "image": lf.stem,
                    "kept": False, "reason": "excluded_species", "boxes_out": 0,
                })
                continue

            # Remap species IDs and write output
            out_lines = []
            for sp_id, cx, cy, w, h, mo_id in boxes:
                new_sp = SPECIES_REMAP[sp_id]
                out_lines.append(f"{new_sp} {cx} {cy} {w} {h} {mo_id}")
                stats["species_counts"][new_sp] += 1
                stats["motion_counts"][mo_id] += 1

            (out_dir / lf.name).write_text("\n".join(out_lines) + "\n")
            kept += 1
            stats["kept"] += 1
            stats["boxes_out"] += len(out_lines)
            summary_rows.append({
                "split": split, "image": lf.stem,
                "kept": True, "reason": "ok", "boxes_out": len(out_lines),
            })

        print(f"{split}: {kept} kept, {dropped_amb} dropped (ambiguous), {dropped_sp} dropped (excluded species)")

    # ── Write data_filtered.yaml ──────────────────────────────────────────────
    yaml_lines = [
        "path: .",
        "train: images_thermal/images/train",
        "val:   images_thermal/images/val",
        "test:  images_thermal/images/test",
        "",
        "# Label format per box: species_id cx cy w h motion_id",
        "# motion: 0=static  1=moving",
        "",
        f"nc: {len(SPECIES_NAMES)}",
        "names:",
    ]
    for name in SPECIES_NAMES:
        yaml_lines.append(f"  - {name}")
    yaml_lines += [
        "",
        "motion_nc: 2",
        "motion_names:",
        "  - static",
        "  - moving",
    ]
    DATA_YAML.write_text("\n".join(yaml_lines), encoding="utf-8")

    # ── Write summary CSV ─────────────────────────────────────────────────────
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "image", "kept", "reason", "boxes_out"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # ── Print stats ───────────────────────────────────────────────────────────
    kept_pct = stats["kept"] / stats["total_in"] * 100 if stats["total_in"] else 0
    print("\n" + "=" * 52)
    print("DONE")
    print("=" * 52)
    print(f"  Images in          : {stats['total_in']}")
    print(f"  Images kept        : {stats['kept']}  ({kept_pct:.1f}%)")
    print(f"  Dropped ambiguous  : {stats['dropped_ambiguous']}")
    print(f"  Dropped species    : {stats['dropped_species']}")
    print(f"  Boxes in           : {stats['boxes_in']}")
    print(f"  Boxes out          : {stats['boxes_out']}")
    print("\nSpecies distribution (filtered):")
    for sp_id, name in enumerate(SPECIES_NAMES):
        count = stats["species_counts"].get(sp_id, 0)
        print(f"  {sp_id}  {name:<20s} {count:6d}")
    print("\nMotion distribution (filtered):")
    for mo_id, name in MOTION_NAMES.items():
        count = stats["motion_counts"].get(mo_id, 0)
        print(f"  {mo_id}  {name:<12s} {count:6d}")
    print(f"\nOutput labels  : {OUTPUT_LABELS_DIR}/")
    print(f"data.yaml      : {DATA_YAML}")
    print(f"Summary CSV    : {SUMMARY_CSV}")


if __name__ == "__main__":
    main()