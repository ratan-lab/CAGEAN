#!/bin/bash
#SBATCH --job-name=multicell
#SBATCH --account=<your_account>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/multicell_%j.log

# CHANGE: activate your conda environment (module load may vary by cluster)
# module load miniforge
conda activate cagean

REPO=$(dirname "$(dirname "$(realpath "$0")")")
MARK=h3k4me3      # CHANGE: h3k4me3 | h3k27ac | atac
OUT=checkpoints/multicell_cagean_${MARK}.pt

mkdir -p logs checkpoints

# CHANGE: set paths to each cell line's numpy arrays
A549=data/a549
HELAS3=data/helas3
HEPG2=data/hepg2

python "${REPO}/train_cagean.py" \
    --cells a549:${A549}:0.0408 helas3:${HELAS3}:auto hepg2:${HEPG2}:0.5 \
    --mark     ${MARK} \
    --val_cell a549 \
    --out      ${OUT} \
    --epochs   50 \
    --patience 7 \
    --batch    256 \
    --lr       1e-3 \
    --min_lr   1e-5 \
    --weight_decay 0.01
