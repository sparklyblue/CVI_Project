"""
BAMBI Label Spot-Checker - Claude generated just for looking
=========================
A local web app to visually verify that species + motion labels
are correctly assigned to image crops.

Usage:
    pip install flask pillow
    python check_labels.py

Then open http://localhost:5001 in your browser.

Shows random samples filtered by species and/or motion,
with bounding boxes drawn and labels overlaid.
"""

import io
import os
import random
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string, request

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow not found. Run: pip install pillow")
    raise

# ── CONFIG ────────────────────────────────────────────────────────────────────

IMAGES_DIR      = Path("images_thermal/images")
LABELS_DIR      = Path("labels_filtered")
SPLITS          = ["train", "val", "test"]

SPECIES_NAMES = {
    0:  "Roe deer",
    1:  "Red deer",
    2:  "Fallow Deer",
    3:  "Wild boar",
    4:  "Alpine ibex",
    5:  "Chamois",
    6:  "Hybrid Pig",
    7:  "Human",
    8:  "Bird",
    9:  "Dog",
    10: "Unknown",
    11: "No-animal",
}

MOTION_NAMES = {0: "static", 1: "moving", 2: "ambiguous"}

# Colours per species (RGB)
SPECIES_COLORS = {
    0:  (30,  158, 117),   # teal      - Roe deer
    1:  (55,  138, 221),   # blue      - Red deer
    2:  (99,  153,  34),   # green     - Fallow Deer
    3:  (216,  90,  48),   # coral     - Wild boar
    4:  (127, 119, 221),   # purple    - Alpine ibex
    5:  (212,  83, 126),   # pink      - Chamois
    6:  (186, 117,  23),   # amber     - Hybrid Pig
    7:  (136, 135, 128),   # gray      - Human
    8:  (29,  158, 165),   # cyan      - Bird
    9:  (226,  75,  74),   # red       - Dog
    10: (180, 178, 169),   # light gray- Unknown
    11: (80,   78,  74),   # dark gray - No-animal
}

MOTION_BORDER = {
    0: (30,  158, 117),    # green  - static
    1: (216,  90,  48),    # orange - moving
    2: (180, 178, 169),    # gray   - ambiguous
}

app = Flask(__name__)

# ── Index: scan all label files once at startup ───────────────────────────────

index = []   # list of dicts: {split, stem, image_path, label_path}

def build_index():
    global index
    index = []
    for split in SPLITS:
        ldir = LABELS_DIR / split
        idir = IMAGES_DIR / split
        if not ldir.exists():
            continue
        for lf in ldir.glob("*.txt"):
            # find matching image
            img = None
            for ext in [".jpg", ".jpeg", ".png"]:
                p = idir / (lf.stem + ext)
                if p.exists():
                    img = p
                    break
            if img:
                index.append({
                    "split": split,
                    "stem": lf.stem,
                    "image_path": str(img),
                    "label_path": str(lf),
                })
    print(f"Index built: {len(index)} image-label pairs")

build_index()

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_labels(label_path):
    """Returns list of (species_id, cx, cy, w, h, motion_id)."""
    boxes = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            sp  = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            mo  = int(parts[5])
            boxes.append((sp, cx, cy, bw, bh, mo))
    return boxes


def draw_image(image_path, boxes, highlight_idx=None):
    """Draw bounding boxes on image, return JPEG bytes."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    iw, ih = img.size

    for i, (sp, cx, cy, bw, bh, mo) in enumerate(boxes):
        x1 = int((cx - bw / 2) * iw)
        y1 = int((cy - bh / 2) * ih)
        x2 = int((cx + bw / 2) * iw)
        y2 = int((cy + bh / 2) * ih)

        color = SPECIES_COLORS.get(sp, (200, 200, 200))
        thickness = 3 if highlight_idx is None or i == highlight_idx else 1
        alpha = 255 if highlight_idx is None or i == highlight_idx else 120

        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

        # Label tag
        sp_name  = SPECIES_NAMES.get(sp, f"cls{sp}")
        mo_name  = MOTION_NAMES.get(mo, f"mo{mo}")
        label    = f"{sp_name} | {mo_name}"
        tag_y    = max(0, y1 - 18)
        draw.rectangle([x1, tag_y, x1 + len(label) * 7 + 6, tag_y + 16], fill=color)
        draw.text((x1 + 3, tag_y + 1), label, fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    buf.seek(0)
    return buf.read()


def filter_index(species_filter, motion_filter, split_filter):
    """Return subset of index matching filters."""
    result = []
    for item in index:
        if split_filter != "all" and item["split"] != split_filter:
            continue
        boxes = read_labels(item["label_path"])
        if not boxes:
            continue
        # Check if any box matches both filters
        match = False
        for sp, cx, cy, bw, bh, mo in boxes:
            sp_ok = (species_filter == -1 or sp == species_filter)
            mo_ok = (motion_filter  == -1 or mo == motion_filter)
            if sp_ok and mo_ok:
                match = True
                break
        if match:
            result.append(item)
    return result

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/sample")
def sample():
    species_filter = request.args.get("species", -1, type=int)
    motion_filter  = request.args.get("motion",  -1, type=int)
    split_filter   = request.args.get("split",  "all")
    n              = request.args.get("n", 12, type=int)

    pool = filter_index(species_filter, motion_filter, split_filter)
    chosen = random.sample(pool, min(n, len(pool)))

    results = []
    for item in chosen:
        boxes = read_labels(item["label_path"])
        results.append({
            "stem":    item["stem"],
            "split":   item["split"],
            "boxes":   [
                {
                    "species":      b[0],
                    "species_name": SPECIES_NAMES.get(b[0], "?"),
                    "motion":       b[5],
                    "motion_name":  MOTION_NAMES.get(b[5], "?"),
                }
                for b in boxes
            ],
        })

    return jsonify({
        "pool_size": len(pool),
        "samples":   results,
    })


@app.route("/image/<stem>")
def serve_image(stem):
    # Find item
    item = next((x for x in index if x["stem"] == stem), None)
    if not item:
        return "Not found", 404
    boxes = read_labels(item["label_path"])
    jpeg  = draw_image(item["image_path"], boxes)
    return Response(jpeg, mimetype="image/jpeg")


@app.route("/stats")
def stats():
    species_counts = {}
    motion_counts  = {}
    split_counts   = {}
    for item in index:
        s = item["split"]
        split_counts[s] = split_counts.get(s, 0) + 1
        boxes = read_labels(item["label_path"])
        for sp, cx, cy, bw, bh, mo in boxes:
            species_counts[sp] = species_counts.get(sp, 0) + 1
            motion_counts[mo]  = motion_counts.get(mo,  0) + 1

    return jsonify({
        "total_images": len(index),
        "splits": split_counts,
        "species": {
            SPECIES_NAMES.get(k, str(k)): v
            for k, v in sorted(species_counts.items())
        },
        "motion": {
            MOTION_NAMES.get(k, str(k)): v
            for k, v in sorted(motion_counts.items())
        },
    })


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BAMBI Label Checker</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#f5f5f3;color:#2c2c2a}
.header{background:#fff;border-bottom:1px solid #e0dfd7;padding:12px 20px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.header h1{font-size:16px;font-weight:500}
.filters{display:flex;gap:10px;flex-wrap:wrap;flex:1}
select,button{padding:6px 12px;border:1px solid #d3d1c7;border-radius:8px;background:#fff;font-size:13px;color:#2c2c2a;cursor:pointer}
button.primary{background:#1D9E75;border-color:#1D9E75;color:#fff}
button.primary:hover{background:#0F6E56}
.stats-bar{background:#fff;border-bottom:1px solid #e0dfd7;padding:8px 20px;font-size:12px;color:#888;display:flex;gap:16px;flex-wrap:wrap}
.stat-item span{font-weight:500;color:#2c2c2a}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;padding:16px}
.card{background:#fff;border:1px solid #e0dfd7;border-radius:10px;overflow:hidden}
.card img{width:100%;display:block;cursor:zoom-in}
.card-body{padding:10px 12px}
.card-title{font-size:12px;font-weight:500;margin-bottom:6px;color:#444}
.tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:12px;margin:2px}
.tag-sp{background:#E1F5EE;color:#0F6E56}
.tag-mo-0{background:#E6F1FB;color:#0C447C}
.tag-mo-1{background:#FAECE7;color:#993C1D}
.tag-mo-2{background:#f0efe8;color:#888}
.loading{text-align:center;padding:60px;color:#888;font-size:14px}
.pool-info{font-size:12px;color:#888;padding:0 20px 0}
.modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:99;align-items:center;justify-content:center}
.modal.open{display:flex}
.modal img{max-width:90vw;max-height:90vh;border-radius:8px}
.modal-close{position:fixed;top:16px;right:20px;color:#fff;font-size:24px;cursor:pointer;background:none;border:none}
</style>
</head>
<body>

<div class="header">
  <h1>BAMBI Label Checker</h1>
  <div class="filters">
    <select id="sel-species">
      <option value="-1">All species</option>
      <option value="0">Roe deer</option>
      <option value="1">Red deer</option>
      <option value="2">Fallow Deer</option>
      <option value="3">Wild boar</option>
      <option value="4">Alpine ibex</option>
      <option value="5">Chamois</option>
      <option value="6">Hybrid Pig</option>
      <option value="7">Human</option>
      <option value="8">Bird</option>
      <option value="9">Dog</option>
      <option value="10">Unknown</option>
      <option value="11">No-animal</option>
    </select>
    <select id="sel-motion">
      <option value="-1">All motion</option>
      <option value="0">Static</option>
      <option value="1">Moving</option>
      <option value="2">Ambiguous</option>
    </select>
    <select id="sel-split">
      <option value="all">All splits</option>
      <option value="train">Train</option>
      <option value="val">Val</option>
      <option value="test">Test</option>
    </select>
    <select id="sel-n">
      <option value="12">12 samples</option>
      <option value="24">24 samples</option>
      <option value="48">48 samples</option>
    </select>
    <button class="primary" onclick="load()">Refresh sample</button>
  </div>
</div>

<div class="stats-bar" id="stats-bar">Loading stats...</div>
<div class="pool-info" id="pool-info"></div>
<div class="grid" id="grid"><div class="loading">Loading...</div></div>

<div class="modal" id="modal" onclick="closeModal()">
  <button class="modal-close" onclick="closeModal()">✕</button>
  <img id="modal-img" src="" alt="full size">
</div>

<script>
function load(){
  const sp = document.getElementById('sel-species').value;
  const mo = document.getElementById('sel-motion').value;
  const split = document.getElementById('sel-split').value;
  const n  = document.getElementById('sel-n').value;
  document.getElementById('grid').innerHTML='<div class="loading">Loading...</div>';
  fetch(`/sample?species=${sp}&motion=${mo}&split=${split}&n=${n}`)
    .then(r=>r.json()).then(data=>{
      document.getElementById('pool-info').textContent=
        `Showing ${data.samples.length} of ${data.pool_size} matching images`;
      const grid=document.getElementById('grid');
      grid.innerHTML='';
      data.samples.forEach(s=>{
        const tags=s.boxes.map(b=>
          `<span class="tag tag-sp">${b.species_name}</span>`+
          `<span class="tag tag-mo-${b.motion}">${b.motion_name}</span>`
        ).join(' ');
        const card=document.createElement('div');
        card.className='card';
        card.innerHTML=`
          <img src="/image/${s.stem}" loading="lazy" onclick="openModal('/image/${s.stem}')" alt="${s.stem}">
          <div class="card-body">
            <div class="card-title">${s.stem} <span style="color:#aaa">[${s.split}]</span></div>
            <div>${tags}</div>
          </div>`;
        grid.appendChild(card);
      });
    });
}

function loadStats(){
  fetch('/stats').then(r=>r.json()).then(d=>{
    const sp=Object.entries(d.species).map(([k,v])=>`<span>${k}: <span>${v}</span></span>`).join('');
    const mo=Object.entries(d.motion).map(([k,v])=>`<span>${k}: <span>${v}</span></span>`).join('');
    document.getElementById('stats-bar').innerHTML=
      `<strong style="color:#2c2c2a">${d.total_images} images</strong> &nbsp;|&nbsp; ${sp} &nbsp;|&nbsp; ${mo}`;
  });
}

function openModal(src){
  document.getElementById('modal-img').src=src;
  document.getElementById('modal').classList.add('open');
}
function closeModal(){
  document.getElementById('modal').classList.remove('open');
  document.getElementById('modal-img').src='';
}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal();});

loadStats();
load();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("\n  BAMBI Label Spot-Checker")
    print("  Open http://localhost:5001\n")
    app.run(port=5001, debug=False)
"""
"""