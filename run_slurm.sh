#!/bin/bash
# ===================================================================
# run_slurm.sh — submit the pipeline as a SLURM job
#
# Usage:
#   ./run_slurm.sh path/to/scene.tif results/scene mine
#   ./run_slurm.sh path/to/scene.tif results/scene auto
# ===================================================================

INPUT="${1:?Usage: ./run_slurm.sh <input.tif> <output_dir> <scene_type>}"
OUTPUT="${2:?Usage: ./run_slurm.sh <input.tif> <output_dir> <scene_type>}"
SCENE_TYPE="${3:-port}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JOB=$(sbatch --parsable << SLURM
#!/bin/bash
#SBATCH --job-name=sar_seg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=64G
#SBATCH --time=50:00:00
#SBATCH --output=$SCRIPT_DIR/logs/sar_seg_%j.txt

source /home/d/domksj/miniconda3/etc/profile.d/conda.sh
conda activate segearth
cd $SCRIPT_DIR

python run_single_image.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --scene-type "$SCENE_TYPE"
SLURM
)

mkdir -p "$SCRIPT_DIR/logs"
echo "Submitted job $JOB"
echo "Log: $SCRIPT_DIR/logs/sar_seg_${JOB}.txt"
echo ""
echo "Monitor with:  tail -f $SCRIPT_DIR/logs/sar_seg_${JOB}.txt"
echo "Check status:  squeue -j $JOB"
