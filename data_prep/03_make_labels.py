"""
Extract G4P ChIP-seq signal over PQS windows and write label arrays.

For each chromosome writes:
  {out_dir}/{chrom}_labels.npy  (N,) float32  raw G4P signal per window

Labels are continuous signal values; binarization at the Q90 threshold is
done at training/evaluation time. This script writes raw signal so the same
arrays can be used with any threshold.

For G4P data (A549, HEK293T): signal is the mean per-base signal p-value
extracted from the G4P BigWig over each 1000-bp window.
For BG4 data (K562): signal is typically a peak score from a narrow-peak BED.

Usage — G4P BigWig:
  python data_prep/03_make_labels.py \\
      --pqs_bed  pqs/PQS_padded.bed \\
      --bigwig   /path/to/g4p_signal.bigwig \\
      --out_dir  data/a549

Usage — BG4 narrow-peak BED (binary):
  python data_prep/03_make_labels.py \\
      --pqs_bed  pqs/PQS_padded.bed \\
      --bed      /path/to/k562_bg4_peaks.bed \\
      --out_dir  data/k562_bg4

Requires: pyBigWig (for --bigwig mode), pybedtools (for --bed mode)
"""
import os
import argparse
import numpy as np
from collections import defaultdict


def _from_bigwig(pqs_bed, bigwig_path, out_dir):
    import pyBigWig
    bw = pyBigWig.open(bigwig_path)

    windows = defaultdict(list)
    with open(pqs_bed) as fh:
        for line in fh:
            p = line.split("\t")
            windows[p[0]].append((int(p[1]), int(p[2])))

    for chrom in sorted(windows):
        wins = sorted(windows[chrom])
        labels = np.zeros(len(wins), dtype=np.float32)
        for i, (start, end) in enumerate(wins):
            try:
                vals = bw.values(chrom, start, end, numpy=True)
                if vals is not None:
                    labels[i] = float(np.nanmean(vals))
            except RuntimeError:
                pass
        np.save(os.path.join(out_dir, f"{chrom}_labels.npy"), labels)
        n_pos_q90 = int((labels > np.percentile(labels[labels > 0], 90)).sum()) \
                    if (labels > 0).any() else 0
        print(f"  {chrom}: {len(wins):,}  signal_max={labels.max():.4f}  "
              f"pos@Q90≈{n_pos_q90:,}")

    bw.close()


def _from_bed(pqs_bed, peaks_bed, out_dir):
    import pybedtools
    pqs_bt  = pybedtools.BedTool(pqs_bed)
    peak_bt = pybedtools.BedTool(peaks_bed)
    hit_ids = {f.name for f in pqs_bt.intersect(peak_bt, u=True)}

    windows = defaultdict(list)
    pqs_ids = defaultdict(list)
    with open(pqs_bed) as fh:
        for line in fh:
            p = line.split("\t")
            windows[p[0]].append((int(p[1]), int(p[2])))
            pqs_ids[p[0]].append(p[3])

    for chrom in sorted(windows):
        ids    = pqs_ids[chrom]
        labels = np.array([1.0 if pid in hit_ids else 0.0 for pid in ids], dtype=np.float32)
        np.save(os.path.join(out_dir, f"{chrom}_labels.npy"), labels)
        print(f"  {chrom}: {len(ids):,}  pos={int(labels.sum()):,} ({labels.mean()*100:.1f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pqs_bed", required=True)
    p.add_argument("--out_dir", required=True)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--bigwig", help="G4P signal BigWig")
    grp.add_argument("--bed",    help="BG4 narrow-peak BED (binary)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.bigwig:
        print(f"Extracting G4P signal from {args.bigwig}...")
        _from_bigwig(args.pqs_bed, args.bigwig, args.out_dir)
    else:
        print(f"Extracting BG4 binary labels from {args.bed}...")
        _from_bed(args.pqs_bed, args.bed, args.out_dir)

    print(f"Done → {args.out_dir}")


if __name__ == "__main__":
    main()
