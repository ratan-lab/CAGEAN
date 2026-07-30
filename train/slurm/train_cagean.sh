#!/bin/bash
#SBATCH --job-name=cagean
#SBATCH --account=<your_account>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cagean_%j.log

# CHANGE: activate your conda environment (module load may vary by cluster)
# module load miniforge
conda activate cagean

REPO=$(dirname "$(dirname "$(realpath "$0")")")
DATA=data/a549    # CHANGE: path to A549 numpy arrays
MARK=h3k4me3      # CHANGE: h3k4me3 | h3k27ac | atac
OUT=checkpoints/cagean_${MARK}.pt

mkdir -p logs checkpoints

python "${REPO}/train_cagean.py" \
    --cells  a549:${DATA}:0.0408 \
    --mark   ${MARK} \
    --out    ${OUT} \
    --epochs 50 \
    --patience 7 \
    --batch  256 \
    --lr     1e-3 \
    --min_lr 1e-5 \
    --weight_decay 0.01
