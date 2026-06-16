#!/bin/bash
# ===================================================================
# run_batch.sh
# Run the single-image SAR segmentation pipeline over many .tif files.
#
# Usage:
#   ./run_batch.sh  <input_dir>  <output_dir>  [scene_type]
#
#   input_dir   folder containing .tif SAR scenes
#   output_dir  where results are written (one subfolder per scene)
#   scene_type  optional: mine | port | urban | auto   (default: auto)
#
# Example:
#   ./run_batch.sh ./input_scenes ./results auto
# ===================================================================

set -euo pipefail

INPUT_DIR="${1:?Usage: ./run_batch.sh <input_dir> <output_dir> [scene_type]}"
OUTPUT_DIR="${2:?Usage: ./run_batch.sh <input_dir> <output_dir> [scene_type]}"
SCENE_TYPE="${3:-auto}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
tifs=("$INPUT_DIR"/*.tif "$INPUT_DIR"/*.tiff)

if [ ${#tifs[@]} -eq 0 ]; then
    echo "No .tif files found in $INPUT_DIR"
    exit 1
fi

echo "Found ${#tifs[@]} scene(s). Scene type: $SCENE_TYPE"
echo "==========================================================="

for tif in "${tifs[@]}"; do
    base="$(basename "$tif")"
    name="${base%.*}"
    echo ""
    echo ">>> [$((++i))/${#tifs[@]}] $name"
    python3 "$SCRIPT_DIR/run_single_image.py" \
        --input "$tif" \
        --output "$OUTPUT_DIR/$name" \
        --scene-type "$SCENE_TYPE"
done

echo ""
echo "==========================================================="
echo "All ${#tifs[@]} scene(s) complete. Results in: $OUTPUT_DIR"
