"""
Zero-epi inference ablation (Supplementary Table S5).

Zeros the epigenomic input to a trained CAGEAN at inference time and
measures the resulting AUPRC. This demonstrates that CAGEAN+zero_epi is
not a valid seq-only baseline: the model receives an input distribution it
never saw during training, producing unpredictable behaviour. In practice
it scores above the seq-only CNN for some marks and below for others,
making it uninterpretable for Δ(arch)/Δ(epi) decomposition.

U2OS uses binary narrowPeak labels with no T+NZ filter. Its seq-only CNN
baseline (Table S5) comes from eval_tf_baseline.py with --ckpt_seqonly and
the U2OS binary-label test set.

Compare the output of this script against:
  - eval_samecell.py      (full epi CAGEAN = upper bound)
  - eval_tf_baseline.py   (seq-only CNN = architecture reference)

Usage:
  python eval/eval_zeroepi.py \\
      --ckpt          checkpoints/cagean_h3k4me3.pt \\
      --data_dir      data/a549/test \\
      --crosscell_dir data/crosscell/hek293t \\
      --u2os_dir      data/crosscell/u2os_bg4 \\
      --pqs_bed       pqs/PQS_padded.bed \\
      --mark          h3k4me3
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
Q_A549      = 4.08277148e-02
Q_HEK       = 0.03790


def infer_zeroepi(model, seqs, device, batch=512):
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            s = torch.tensor(seqs[i:i+batch], dtype=torch.float32).to(device)
            e = torch.zeros(len(s), 1, 1000, dtype=torch.float32).to(device)
            preds.append(model(torch.cat([s, e], dim=1)).reshape(-1).cpu().numpy())
    return np.concatenate(preds)


def _build_pqs_chr(pqs_bed):
    pqs_chr = {}
    with open(pqs_bed) as fh:
        for line in fh:
            p = line.split("\t", 5)
            pqs_chr[p[3]] = p[0]
    return pqs_chr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          required=True)
    p.add_argument("--data_dir",      required=True, help="A549 test chromosome arrays")
    p.add_argument("--crosscell_dir", required=True, help="HEK293T arrays directory")
    p.add_argument("--u2os_dir",      default=None,  help="U2OS BG4 arrays directory (optional)")
    p.add_argument("--pqs_bed",       required=True)
    p.add_argument("--mark",          required=True, choices=["h3k4me3", "h3k27ac", "atac"])
    p.add_argument("--batch",         default=512, type=int)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CAGEAN()
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval().to(device)
    print(f"CAGEAN+zero_epi  mark={args.mark}  device={device}")
    print("NOTE: zeroed epi produces OOD input; results are for illustration only.")

    # A549 same-cell
    all_seqs, all_labels = [], []
    for chrn in TEST_CHROMS:
        all_seqs.append(  np.load(os.path.join(args.data_dir, f"{chrn}_seqs.npy")))
        all_labels.append(np.load(os.path.join(args.data_dir, f"{chrn}_labels.npy")))
    seqs       = np.concatenate(all_seqs)
    labels_bin = (np.concatenate(all_labels) > Q_A549).astype(np.int32)
    preds      = infer_zeroepi(model, seqs, device, args.batch)
    auprc_a549 = average_precision_score(labels_bin, preds)
    print(f"\nA549  CAGEAN+zero_epi AUPRC = {auprc_a549:.4f}")

    # HEK293T cross-cell T+NZ
    # pqs_chr is genome-wide (from PQS_padded.bed), valid for all cell types below.
    pqs_chr    = _build_pqs_chr(args.pqs_bed)
    labels_all = np.load(os.path.join(args.crosscell_dir, "HEK293T_labels.npy"))
    with open(os.path.join(args.crosscell_dir, "HEK293T_pqs_ids.txt")) as fh:
        pqs_ids = fh.read().split()
    test_set   = set(TEST_CHROMS)
    test_mask  = np.array([pqs_chr.get(p, "") in test_set for p in pqs_ids])
    tnz_mask   = test_mask & (labels_all > 0)
    lb_tnz     = (labels_all[tnz_mask] > Q_HEK).astype(np.int32)
    seqs_tnz   = np.load(os.path.join(args.crosscell_dir, "HEK293T_seqs.npy"))[tnz_mask]
    preds      = infer_zeroepi(model, seqs_tnz, device, args.batch)
    auprc_hek  = average_precision_score(lb_tnz, preds)
    print(f"HEK293T  CAGEAN+zero_epi AUPRC = {auprc_hek:.4f}")

    # U2OS BG4 cross-cell+technique (binary labels, no T+NZ)
    if args.u2os_dir:
        labels_u2os = np.load(os.path.join(args.u2os_dir, "U2OS_bg4_labels.npy"))
        with open(os.path.join(args.u2os_dir, "U2OS_bg4_pqs_ids.txt")) as fh:
            pqs_u2os = fh.read().split()
        seqs_u2os_full = np.load(os.path.join(args.u2os_dir, "U2OS_bg4_seqs.npy"), mmap_mode="r")
        if not (len(labels_u2os) == len(pqs_u2os) == len(seqs_u2os_full)):
            raise RuntimeError(
                f"U2OS array length mismatch: labels={len(labels_u2os)}, "
                f"pqs_ids={len(pqs_u2os)}, seqs={len(seqs_u2os_full)}"
            )
        test_mask_u = np.array([pqs_chr.get(p, "") in test_set for p in pqs_u2os])
        lb_u2os     = labels_u2os[test_mask_u].astype(np.int32)
        seqs_u2os   = seqs_u2os_full[test_mask_u].copy()
        preds_u2os  = infer_zeroepi(model, seqs_u2os, device, args.batch)
        auprc_u2os  = average_precision_score(lb_u2os, preds_u2os)
        print(f"U2OS BG4  CAGEAN+zero_epi AUPRC = {auprc_u2os:.4f}")


if __name__ == "__main__":
    main()
