"""
Extract epigenomic signal and one-hot DNA sequence over each PQS window.

For each chromosome, writes:
  {out_dir}/{chrom}_seqs.npy       (N, 4, 1000) int8  one-hot (A,T,C,G order)
  {out_dir}/{chrom}_epi_{mark}.npy (N, 1000)    float32  BigWig signal
  {out_dir}/{chrom}_covered_{mark}.npy (N,)     bool   sites with non-zero signal

One-hot channel order: 0=A, 1=T, 2=C, 3=G  (ATCG)

Usage:
  python data_prep/02_extract_epi_signal.py \\
      --pqs_bed   pqs/PQS_padded.bed \\
      --genome    hg19.fa \\
      --bigwig    /path/to/h3k4me3.bigwig \\
      --mark      h3k4me3 \\
      --out_dir   data/a549 \\
      --chroms    chr1 chr2   [optional; default: all in PQS BED]

Requires: pyBigWig, biopython
"""
import os
import argparse
import numpy as np
import pyBigWig
from Bio import SeqIO
from collections import defaultdict

BASE_TO_IDX = {"A": 0, "T": 1, "C": 2, "G": 3}


def one_hot(seq: str) -> np.ndarray:
    arr = np.zeros((4, len(seq)), dtype=np.int8)
    for i, b in enumerate(seq.upper()):
        idx = BASE_TO_IDX.get(b)
        if idx is not None:
            arr[idx, i] = 1
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pqs_bed",  required=True)
    p.add_argument("--genome",   required=True, help="hg19 FASTA")
    p.add_argument("--bigwig",   required=True, help="Signal BigWig file")
    p.add_argument("--mark",     required=True, help="Mark name for output filename")
    p.add_argument("--out_dir",  required=True)
    p.add_argument("--chroms",   nargs="*", default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Group PQS windows by chromosome
    windows = defaultdict(list)
    with open(args.pqs_bed) as fh:
        for line in fh:
            parts = line.split("\t")
            chrom, start, end, pqs_id = parts[0], int(parts[1]), int(parts[2]), parts[3]
            windows[chrom].append((start, end, pqs_id))

    keep_chroms = set(args.chroms) if args.chroms else set(windows)

    print(f"Loading genome from {args.genome}...")
    genome = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(args.genome, "fasta")
              if rec.id in keep_chroms}

    print(f"Opening BigWig: {args.bigwig}")
    bw = pyBigWig.open(args.bigwig)

    for chrom in sorted(keep_chroms):
        if chrom not in windows:
            continue
        wins = sorted(windows[chrom], key=lambda x: x[0])
        n    = len(wins)
        seq_arr = np.zeros((n, 4, 1000), dtype=np.int8)
        epi_arr = np.zeros((n, 1000),    dtype=np.float32)

        chrom_seq = genome.get(chrom, "")
        if not chrom_seq:
            print(f"  {chrom}: genome sequence not found, skipping")
            continue

        for i, (start, end, _) in enumerate(wins):
            # Sequence
            sub = chrom_seq[start:end]
            if len(sub) == 1000:
                seq_arr[i] = one_hot(sub)
            # Epigenomic signal — fill NaN with 0
            try:
                vals = bw.values(chrom, start, end, numpy=True)
                if vals is not None:
                    vals = np.nan_to_num(vals, nan=0.0)
                    epi_arr[i] = vals.astype(np.float32)
            except RuntimeError:
                pass  # region not in BigWig → zeros

        covered = (epi_arr.sum(axis=1) > 0)

        np.save(os.path.join(args.out_dir, f"{chrom}_seqs.npy"),               seq_arr)
        np.save(os.path.join(args.out_dir, f"{chrom}_epi_{args.mark}.npy"),    epi_arr)
        np.save(os.path.join(args.out_dir, f"{chrom}_covered_{args.mark}.npy"), covered)
        print(f"  {chrom}: {n:,} windows  covered={covered.sum():,} ({covered.mean()*100:.1f}%)")

    bw.close()
    print(f"Done → {args.out_dir}")


if __name__ == "__main__":
    main()
