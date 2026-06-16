#!/bin/bash
# ===================================================================
# setup.sh — create the conda environment for the SAR pipeline
# ===================================================================
set -euo pipefail

ENV_NAME="sar-seg"

echo "Creating conda environment '$ENV_NAME'..."

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create from environment.yml if present, else inline
if [ -f environment.yml ]; then
    conda env create -f environment.yml -n "$ENV_NAME" || \
        conda env update -f environment.yml -n "$ENV_NAME"
else
    conda create -y -n "$ENV_NAME" python=3.10
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    pip install \
        "torch>=2.0" torchvision \
        numpy pillow rasterio scipy scikit-image \
        open_clip_torch ftfy regex tqdm
fi

echo ""
echo "==========================================================="
echo "Done. Activate with:   conda activate $ENV_NAME"
echo ""
echo "Next steps:"
echo "  1. Copy model weights into  models/  (see README.md)"
echo "  2. Put a .tif into          input_scenes/"
echo "  3. Run: python run_single_image.py --input input_scenes/X.tif \\"
echo "             --output results/X --scene-type mine"
echo "==========================================================="
