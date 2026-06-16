#!/bin/bash
# ===================================================================
# run.sh — wrapper that activates the conda env then runs the pipeline
#
# Usage:
#   ./run.sh --input path/to/scene.tif --output results/scene --scene-type mine
# ===================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Use segearth env (has AlignEarth + torch); install any missing deps
conda activate segearth

# Ensure all required packages are present
python -c "from skimage.morphology import disk" 2>/dev/null || \
    pip install scikit-image --quiet

python -c "import rasterio" 2>/dev/null || \
    pip install rasterio --quiet

python -c "import scipy" 2>/dev/null || \
    pip install scipy --quiet

# Run the pipeline
cd "$SCRIPT_DIR"
python run_single_image.py "$@"
