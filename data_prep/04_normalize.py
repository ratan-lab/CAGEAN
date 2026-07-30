"""
Z-normalize epigenomic signal using training-chromosome statistics.

Computes per-mark mean and variance from the training chromosomes of the
source cell type (A549), then applies the same transformation to all
chromosomes of all cell types. Saves normalization parameters to a JSON
file so the same statistics can be applied consistently at evaluation time
and distributed alongside model weights (see README, Zenodo section).

NOTE: CAGEAN uses InstanceNorm1d in its epigenomic stem, which normalizes
each sample independently at inference time. Z-normalization at the data
level is therefore redundant for CAGEAN and is NOT applied to CAGEAN
inputs in this pipeline. This script exists to support the epiG4NN TF
baseline, whose batch normalization uses training-set running statistics
and requires pre-normalized input to perform cross-cell evaluation fairly.

Usage:
  python data_prep/04_normalize.py \\
      --a549_train_dir data/a549/train \\
      --mark h3k4me3 \\
      --params_out data/a549/norm_params_h3k4me3.json \\
      --apply_to   data/hek293t data/k562_bg4
"""
import os
import json
import argparse
import numpy as np
from glob import glob

TRAIN_CHROMS = [
    "chr2", "chr4", "chr6", "chr8", "chr11", "chr12", "chr13",
    "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
    "chr20", "chr21", "chr22", "chrX", "chrY",
]


def compute_stats(train_dir, mark, max_samples=5_000_000):
    """Compute mean and std from training chromosomes (subsample if large)."""
    arrays = []
    total  = 0
    for chrn in TRAIN_CHROMS:
        path = os.path.join(train_dir, f"{chrn}_epi_{mark}.npy")
        if not os.path.exists(path):
            continue
        arr = np.load(path, mmap_mode="r").ravel()
        arrays.append(arr); total += len(arr)

    target = max_samples
    sampled = []
    for arr in arrays:
        step = max(1, len(arr) // max(1, round(len(arr) / total * target)))
        sampled.append(arr[::step].astype(np.float32))
    flat = np.concatenate(sampled)
    return float(flat.mean()), float(flat.std())


def apply_norm(data_dir, mark, mean, std, suffix="_znorm"):
    """Z-normalize epi arrays in data_dir in-place (writes new files)."""
    for path in sorted(glob(os.path.join(data_dir, f"*_epi_{mark}.npy"))):
        arr  = np.load(path).astype(np.float32)
        znorm = (arr - mean) / (std + 1e-8)
        out_path = path.replace(f"_epi_{mark}.npy", f"_epi_{mark}{suffix}.npy")
        np.save(out_path, znorm)
        print(f"  Wrote {os.path.basename(out_path)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a549_train_dir", required=True,
                   help="A549 training-chromosome arrays")
    p.add_argument("--mark",           required=True)
    p.add_argument("--params_out",     required=True,
                   help="JSON file to save normalization params")
    p.add_argument("--apply_to",       nargs="*", default=[],
                   help="Directories to apply normalization to (writes *_znorm.npy)")
    args = p.parse_args()

    print(f"Computing A549 training stats for {args.mark}...")
    mean, std = compute_stats(args.a549_train_dir, args.mark)
    print(f"  mean={mean:.6f}  std={std:.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.params_out)), exist_ok=True)
    with open(args.params_out, "w") as fh:
        json.dump({"mark": args.mark, "mean": mean, "std": std}, fh, indent=2)
    print(f"Saved params → {args.params_out}")

    for d in args.apply_to:
        print(f"Normalizing {d}...")
        apply_norm(d, args.mark, mean, std)

    print("Done.")


if __name__ == "__main__":
    main()
