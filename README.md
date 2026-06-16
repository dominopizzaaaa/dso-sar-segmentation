# SAR Semantic Segmentation — Portable Package

Self-contained package to run the full SAR segmentation pipeline on a **single
SAR image** with one command. Designed so that a new user can drop in a `.tif`
and get a segmentation map without reconstructing the development environment by
hand.

```
AlignEarth (zero-shot)  →  P10 (physics refinement)  →  P11 (watershed carving)
```

---

## Folder Layout

```
sar-segmentation-portable/
├── README.md                 ← this file
├── run_single_image.py       ← run the pipeline on ONE .tif
├── run_batch.sh              ← loop the runner over many .tif files
├── setup.sh                  ← create the conda env + install deps
├── environment.yml           ← pinned dependencies
├── groq_key.txt              ← (you provide) Groq API key, for --scene-type auto
│
├── docs/
│   ├── PIPELINE.md           ← full pipeline walkthrough with diagrams
│   ├── ASSUMPTIONS.md        ← thresholding + VLM/LLM assumptions
│   └── diagrams/             ← rendered diagrams (png + svg source)
│
├── models/                   ← (you populate) model weights — see below
│   ├── SegEarth-OV-2/        ← AlignEarth codebase + checkpoint
│   └── GeoChat/              ← GeoChat-7B weights (only for --scene-type auto)
│
├── input_scenes/             ← put your .tif SAR images here
└── results/                  ← outputs are written here
```

---

## Quick Start

### 1. Set up the environment (once)

```bash
bash setup.sh
conda activate sar-seg
```

### 2. Add the model weights

The model weights are **not bundled** (they are large and licensed). Copy them
from the cluster into `models/`:

```bash
# AlignEarth (required) — the SegEarth-OV-2 codebase + checkpoint
#   from cluster: ~/sar_data/SegEarth-OV-2/
cp -r /path/to/SegEarth-OV-2 models/

# GeoChat (only needed for --scene-type auto)
#   from cluster: ~/sar_data/GeoChat/
cp -r /path/to/GeoChat models/
```

Expected paths after copying:
```
models/SegEarth-OV-2/checkpoint/AlignEarth-SAR-ViT-B-16.pt
models/GeoChat/weights/GeoChat-7B/
```

### 3. Run on one image

```bash
# Explicit scene type (deterministic, no VLM needed):
python run_single_image.py \
    --input  input_scenes/my_scene.tif \
    --output results/my_scene \
    --scene-type mine          # mine | port | urban

# Or let the VLM decide (needs GeoChat + groq_key.txt):
python run_single_image.py \
    --input  input_scenes/my_scene.tif \
    --output results/my_scene \
    --scene-type auto
```

### 4. Run on a whole folder

```bash
./run_batch.sh  input_scenes/  results/  mine
```

---

## Outputs

For `my_scene.tif` you get, in `results/my_scene/`:

| File | Description |
|------|-------------|
| `my_scene_p8.png` | Raw AlignEarth prediction |
| `my_scene_p10.png` | After P10 refinement |
| `my_scene_final.png` | Final result with building outlines |
| `my_scene_final.npy` | Class array (uint8, 0–5) |

**Class colours:** background=black, building=red, road=yellow,
vegetation=green, water=blue, bareland=brown.

---

## Choosing `--scene-type`

| Mode | Needs | Deterministic? | When to use |
|------|-------|----------------|-------------|
| `mine` / `port` / `urban` | nothing extra | yes | You know the scene type |
| `auto` | GeoChat weights + Groq key | nearly | You want it decided automatically |

If you are unsure of the scene type and don't have the VLM set up, `urban` is
the safest manual default (least aggressive corrections).

---

## Requirements

- Linux with an NVIDIA GPU (CUDA). CPU works but AlignEarth will be very slow.
- conda / miniconda
- `curl` on PATH (used for the Groq call in auto mode)
- ~16 GB GPU memory for large scenes; smaller scenes run in <1 GB.

See `docs/PIPELINE.md` for the full technical walkthrough and
`docs/ASSUMPTIONS.md` for every assumption behind the thresholds and the VLM.

---

## Author

Dominic Koh Song Jun (A0269656J)
DSO National Laboratories, Sensors Division · Supervisor: Peh Ruijie
January – June 2026
