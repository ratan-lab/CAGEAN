"""
Stratified AUPRC on HEK293T cross-cell T+NZ (Table 4).

Partitions test sites into promoter (±2 kb of any TSS), enhancer-like
(per-site mean H3K27ac > Q75 of non-zero per-site means), and inactive.
Promoter takes priority over enhancer when both conditions are met.

Usage:
  python eval/eval_stratified.py \\
      --ckpt_h3k4me3 checkpoints/cagean_h3k4me3.pt \\
      --ckpt_h3k27ac checkpoints/cagean_h3k27ac.pt \\
      --ckpt_atac    checkpoints/cagean_atac.pt \\
      --crosscell_dir data/crosscell/hek293t \\
      --pqs_bed      pqs/PQS_padded.bed \\
      --tss_bed      data/tss_2kb_hg19.bed

CRITICAL: Enhancer threshold is computed from per-site mean H3K27ac signal
(epi.mean(axis=1)), not from the raw 2D array. Computing the percentile on
the raw array before mean-pooling inflates the threshold by ~13% and
misclassifies ~17k sites.  Unit test: threshold should be ≈0.00121 for
HEK293T T+NZ.
"""
import os
import sys
import argparse
import numpy as np
import torch
import pybedtools
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cagean import CAGEAN

TEST_CHROMS = ["chr1", "chr3", "chr5", "chr7", "chr9"]
Q_HEK       = 0.03790
MARKS       = ["h3k4me3", "h3k27ac", "atac"]


def infer(model, seqs, epi, device, batch=512):
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            s = torch.tensor(seqs[i:i+batch], dtype=torch.float32).to(device)
            e = torch.tensor(epi[i:i+batch],  dtype=torch.float32).unsqueeze(1).to(device)
            preds.append(model(torch.cat([s, e], dim=1)).squeeze(-1).cpu().numpy())
    return np.concatenate(preds)


def _build_promoter_set(pqs_bed, tss_bed):
    overlap = pybedtools.BedTool(pqs_bed).intersect(pybedtools.BedTool(tss_bed), u=True)
    return {f.name for f in overlap}


def _build_pqs_chr(pqs_bed):
    pqs_chr = {}
    with open(pqs_bed) as fh:
        for line in fh:
            p = line.split("\t", 5)
            pqs_chr[p[3]] = p[0]
    return pqs_chr


def _report(name, lb, preds):
    pos = lb.mean() * 100
    auprc = average_precision_score(lb, preds) if lb.sum() > 0 else float("nan")
    print(f"  [{name:16s}]  n={len(lb):6,}  pos={pos:4.1f}%  AUPRC={auprc:.4f}")
    return auprc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_h3k4me3",  required=True)
    p.add_argument("--ckpt_h3k27ac",  required=True)
    p.add_argument("--ckpt_atac",     required=True)
    p.add_argument("--crosscell_dir", required=True,
                   help="Directory with HEK293T_*.npy and HEK293T_pqs_ids.txt")
    p.add_argument("--pqs_bed",       required=True)
    p.add_argument("--tss_bed",       required=True,
                   help="TSS ±2 kb BED (hg19); create with data_prep/01_scan_pqs.py steps")
    p.add_argument("--q_hek",         default=Q_HEK, type=float)
    p.add_argument("--batch",         default=512,   type=int)
    args = p.parse_args()

    ckpts  = {"h3k4me3": args.ckpt_h3k4me3,
              "h3k27ac": args.ckpt_h3k27ac,
              "atac":    args.ckpt_atac}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building promoter set...")
    promoter_set = _build_promoter_set(args.pqs_bed, args.tss_bed)
    pqs_chr      = _build_pqs_chr(args.pqs_bed)
    print(f"  Promoter PQS: {len(promoter_set):,}")

    print("Loading HEK293T T+NZ indices...")
    labels_all = np.load(os.path.join(args.crosscell_dir, "HEK293T_labels.npy"))
    pqs_ids    = open(os.path.join(args.crosscell_dir, "HEK293T_pqs_ids.txt")).read().split()
    test_set   = set(TEST_CHROMS)
    test_mask  = np.array([pqs_chr.get(p, "") in test_set for p in pqs_ids])
    tnz_mask   = test_mask & (labels_all > 0)
    print(f"  T+NZ sites: {tnz_mask.sum():,}")

    lb_tnz   = (labels_all[tnz_mask] > args.q_hek).astype(np.int32)
    pqs_tnz  = [p for p, m in zip(pqs_ids, tnz_mask) if m]
    seqs_tnz = np.load(os.path.join(args.crosscell_dir, "HEK293T_seqs.npy"))[tnz_mask]

    # Enhancer threshold from per-site mean H3K27ac (MUST be mean before percentile)
    epi_h3k27ac      = np.load(os.path.join(args.crosscell_dir, "HEK293T_h3k27ac_epi.npy"))[tnz_mask]
    epi_mean_h3k27ac = epi_h3k27ac.mean(axis=1)   # (N,) per-site means — compute FIRST
    epi_nz           = epi_mean_h3k27ac[epi_mean_h3k27ac > 0]
    thresh           = float(np.percentile(epi_nz, 75)) if len(epi_nz) > 0 else 0.0
    print(f"  H3K27ac Q75 of per-site means: {thresh:.5f}  (expected ≈0.00121)")
    del epi_h3k27ac

    # Category assignment: 0=promoter, 1=enhancer, 2=inactive
    cats = np.full(len(seqs_tnz), 2, dtype=np.int8)
    for i, pid in enumerate(pqs_tnz):
        if pid in promoter_set:
            cats[i] = 0
        elif epi_mean_h3k27ac[i] >= thresh:
            cats[i] = 1
    print(f"  Promoter={( cats==0).sum():,}  Enhancer-like={(cats==1).sum():,}  "
          f"Inactive={(cats==2).sum():,}")

    strata = [("ALL", None), ("Promoter", 0), ("Enhancer-like", 1), ("Inactive", 2)]

    for mark in MARKS:
        print(f"\n=== {mark} ===")
        model = CAGEAN()
        model.load_state_dict(
            torch.load(ckpts[mark], map_location=device, weights_only=True))
        model.eval().to(device)

        epi_tnz = np.load(
            os.path.join(args.crosscell_dir, f"HEK293T_{mark}_epi.npy"))[tnz_mask]
        preds = infer(model, seqs_tnz, epi_tnz, device, args.batch)

        for name, cat_id in strata:
            mask = np.ones(len(lb_tnz), dtype=bool) if cat_id is None else (cats == cat_id)
            if mask.sum() >= 50:
                _report(name, lb_tnz[mask], preds[mask])
            else:
                print(f"  [{name:16s}]  n={mask.sum()} — too few to report")

        del model, epi_tnz

    print("\nDone.")


if __name__ == "__main__":
    main()
