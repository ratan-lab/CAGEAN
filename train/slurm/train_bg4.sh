#!/bin/bash
#SBATCH --job-name=bg4_train
#SBATCH --account=<your_account>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bg4_%j.log

# CHANGE: activate your conda environment (module load may vary by cluster)
# module load miniforge
conda activate cagean

REPO=$(dirname "$(dirname "$(realpath "$0")")")
DATA=data/k562_bg4   # CHANGE: path to K562 BG4 numpy arrays
Q=0.01842            # Q90 of non-zero K562 BG4 signal on test chromosomes

mkdir -p logs checkpoints

# Seq-only Transformer
python "${REPO}/train_bg4.py" \
    --model seq_only --data_dir ${DATA} --q ${Q} \
    --out checkpoints/bg4_seqonly_transformer.pt

# CAGEAN (one job per mark; run all three sequentially or submit separate jobs)
for MARK in h3k4me3 h3k27ac atac; do
    python "${REPO}/train_bg4.py" \
        --model cagean --mark ${MARK} --data_dir ${DATA} --q ${Q} \
        --out checkpoints/bg4_cagean_${MARK}.pt
done
