"""
Evaluate CAGEAN on H9 ESC held-out test chromosomes (Tables 5, 6).

Compares single-cell (A549-trained) vs multi-cell (A549+HeLa.S3+HepG2-trained)
CAGEAN on a cell type never seen during training.

Usage:
  python eval/eval_h9esc.py \\
      --ckpt_single   checkpoints/cagean_h3k4me3.pt \\
      --ckpt_multi    checkpoints/multicell_cagean_h3k4me3.pt \\
      --data_dir      data/h9esc/test \\
      --mark          h3k4me3

Data directory layout:
  {data_dir}/{chrom}_seqs.npy
  {data_dir}/{chrom}_labels.npy    (binary 0/1 from BED peaks, threshold q=0.5)
  {data_dir}/{chrom}_epi_{mark}.npy
  {data_dir}/{chrom}_covered_{mark}.npy   [optional]
"""
import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cagean import CAGEAN

TEST_CHROMS = ["chr1", "chr3", "chr5", "chr7", "chr9"]
Q_H9ESC     = 0.5


def infer(model, seqs, epi, device, batch=512):
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            s = torch.tensor(seqs[i:i+batch], dtype=torch.float32).to(device)
            e = torch.tensor(epi[i:i+batch],  dtype=torch.float32).unsqueeze(1).to(device)
            preds.append(model(torch.cat([s, e], dim=1)).reshape(-1).cpu().numpy())
    return np.concatenate(preds)


def _load_h9esc(data_dir, mark, q=Q_H9ESC):
    all_seqs, all_epi, all_labels = [], [], []
    for chrn in TEST_CHROMS:
        seqs   = np.load(os.path.join(data_dir, f"{chrn}_seqs.npy"))
        labels = np.load(os.path.join(data_dir, f"{chrn}_labels.npy"))
        epi    = np.load(os.path.join(data_dir, f"{chrn}_epi_{mark}.npy"))
        cp     = os.path.join(data_dir, f"{chrn}_covered_{mark}.npy")
        if os.path.exists(cp):
            mask = np.load(cp)
            seqs, labels, epi = seqs[mask], labels[mask], epi[mask]
        all_seqs.append(seqs); all_epi.append(epi); all_labels.append(labels)
    seqs_cat   = np.concatenate(all_seqs)
    epi_cat    = np.concatenate(all_epi)
    labels_bin = (np.concatenate(all_labels) > q).astype(np.int32)
    return seqs_cat, epi_cat, labels_bin


def _eval_model(ckpt, seqs, epi, labels, device, batch, label):
    model = CAGEAN()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval().to(device)
    preds = infer(model, seqs, epi, device, batch)
    auprc = average_precision_score(labels, preds)
    print(f"  {label:35s}  AUPRC = {auprc:.4f}")
    del model
    return auprc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_single", required=True,
                   help="Single-cell (A549) CAGEAN checkpoint")
    p.add_argument("--ckpt_multi",  required=True,
                   help="Multi-cell CAGEAN checkpoint")
    p.add_argument("--data_dir",    required=True,
                   help="H9 ESC test chromosome arrays")
    p.add_argument("--mark",        required=True, choices=["h3k4me3", "h3k27ac", "atac"])
    p.add_argument("--q",           default=Q_H9ESC, type=float)
    p.add_argument("--batch",       default=512,     type=int)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"H9 ESC held-out evaluation  mark={args.mark}  device={device}")

    seqs, epi, labels = _load_h9esc(args.data_dir, args.mark, args.q)
    n_pos = int(labels.sum())
    print(f"H9 ESC test: {len(seqs):,} samples  pos={n_pos:,} ({n_pos/len(seqs)*100:.1f}%)")

    _eval_model(args.ckpt_single, seqs, epi, labels, device, args.batch,
                "Single-cell CAGEAN (A549 only)")
    _eval_model(args.ckpt_multi,  seqs, epi, labels, device, args.batch,
                "Multi-cell CAGEAN (A549+HeLa.S3+HepG2)")


if __name__ == "__main__":
    main()
