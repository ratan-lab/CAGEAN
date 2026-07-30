"""
Evaluate seq-only Transformer and CAGEAN on K562 BG4 same-cell (Table 3).

Uses T+NZ protocol: test chromosomes with non-zero signal, Q90 threshold.

Usage:
  python eval/eval_bg4_samecell.py \\
      --ckpt_seqonly checkpoints/bg4_seqonly_transformer.pt \\
      --ckpt_h3k4me3 checkpoints/bg4_cagean_h3k4me3.pt \\
      --ckpt_h3k27ac checkpoints/bg4_cagean_h3k27ac.pt \\
      --ckpt_atac    checkpoints/bg4_cagean_atac.pt \\
      --data_dir     data/k562_bg4

Data directory: per-chromosome arrays (same layout as train/train_bg4.py).
"""
import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cagean import CAGEAN
from models.seqonly_transformer import SeqOnlyTransformer

TEST_CHROMS = ["chr1", "chr3", "chr5", "chr7", "chr9"]
MARKS       = ["h3k4me3", "h3k27ac", "atac"]


def infer_seq(model, seqs, device, batch=512):
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            s = torch.tensor(seqs[i:i+batch], dtype=torch.float32).to(device)
            preds.append(model(s).reshape(-1).cpu().numpy())
    return np.concatenate(preds)


def infer_cagean(model, seqs, epi, device, batch=512):
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            s = torch.tensor(seqs[i:i+batch], dtype=torch.float32).to(device)
            e = torch.tensor(epi[i:i+batch],  dtype=torch.float32).unsqueeze(1).to(device)
            preds.append(model(torch.cat([s, e], dim=1)).reshape(-1).cpu().numpy())
    return np.concatenate(preds)


def _load_test(data_dir, mark=None):
    all_seqs, all_labels = [], []
    all_epi = [] if mark else None
    for chrn in sorted(TEST_CHROMS):
        all_seqs.append(  np.load(os.path.join(data_dir, f"{chrn}_seqs.npy")))
        all_labels.append(np.load(os.path.join(data_dir, f"{chrn}_labels.npy")))
        if mark:
            all_epi.append(np.load(os.path.join(data_dir, f"{chrn}_epi_{mark}.npy")))
    labels_all = np.concatenate(all_labels)
    seqs_all   = np.concatenate(all_seqs)
    epi_all    = np.concatenate(all_epi) if all_epi else None
    return seqs_all, epi_all, labels_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_seqonly", required=True)
    p.add_argument("--ckpt_h3k4me3", required=True)
    p.add_argument("--ckpt_h3k27ac", required=True)
    p.add_argument("--ckpt_atac",    required=True)
    p.add_argument("--data_dir",     required=True)
    p.add_argument("--batch",        default=512, type=int)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts  = {"h3k4me3": args.ckpt_h3k4me3,
              "h3k27ac": args.ckpt_h3k27ac,
              "atac":    args.ckpt_atac}
    print(f"K562 BG4 same-cell evaluation  device={device}")

    # Build T+NZ mask from labels (non-zero = any epi mark present; use seqs-only load)
    seqs_all, _, labels_raw = _load_test(args.data_dir)
    nz_mask = labels_raw > 0
    q90     = float(np.percentile(labels_raw[nz_mask], 90))
    lb_bin  = (labels_raw[nz_mask] > q90).astype(np.int32)
    seqs    = seqs_all[nz_mask]
    print(f"T+NZ: {nz_mask.sum():,}  Q90={q90:.5f}  pos={lb_bin.mean()*100:.1f}%")

    # Seq-only Transformer
    model_seq = SeqOnlyTransformer()
    model_seq.load_state_dict(
        torch.load(args.ckpt_seqonly, map_location=device, weights_only=True))
    model_seq.eval().to(device)
    auprc_seq = average_precision_score(lb_bin, infer_seq(model_seq, seqs, device, args.batch))
    print(f"\nSeq-only Transformer  AUPRC = {auprc_seq:.4f}")
    del model_seq

    # CAGEAN per mark
    for mark in MARKS:
        _, epi_all, _ = _load_test(args.data_dir, mark)
        epi = epi_all[nz_mask]
        model = CAGEAN()
        model.load_state_dict(
            torch.load(ckpts[mark], map_location=device, weights_only=True))
        model.eval().to(device)
        auprc = average_precision_score(lb_bin, infer_cagean(model, seqs, epi, device, args.batch))
        delta_epi = auprc - auprc_seq
        print(f"CAGEAN+{mark:8s}  AUPRC = {auprc:.4f}  Δ(epi)={delta_epi:+.3f}")
        del model, epi_all, epi

    print("\nDone.")


if __name__ == "__main__":
    main()
