"""
Evaluate CAGEAN or SeqOnlyTransformer on HEK293T cross-cell, K562 cross-technique (BG4),
and U2OS cross-cell+technique (BG4).

HEK293T and K562 use the T+NZ protocol: test chromosomes (chr1,3,5,7,9) with
non-zero G4-occupancy signal in the evaluation cell type (labels_all > 0).
U2OS uses binary labels from a narrowPeak consensus (1 = overlaps BG4 peak); all
test-chromosome PQS are evaluated without a non-zero filter.

Usage:
  # CAGEAN
  python eval/eval_crosscell.py \\
      --ckpt          checkpoints/cagean_h3k4me3.pt \\
      --crosscell_dir data/crosscell \\
      --pqs_bed       pqs/PQS_padded.bed \\
      --mark          h3k4me3

  # Seq-only Transformer
  python eval/eval_crosscell.py \\
      --model         seqonly \\
      --ckpt          checkpoints/seqonly_transformer.pt \\
      --crosscell_dir data/crosscell \\
      --pqs_bed       pqs/PQS_padded.bed

Expected layout under --crosscell_dir:
  hek293t/HEK293T_seqs.npy
  hek293t/HEK293T_labels.npy
  hek293t/HEK293T_pqs_ids.txt
  hek293t/HEK293T_{mark}_epi.npy     (only needed for --model cagean)
  k562_bg4/K562_bg4_seqs.npy
  k562_bg4/K562_bg4_labels.npy
  k562_bg4/K562_bg4_pqs_ids.txt
  k562_bg4/K562_bg4_{mark}_epi.npy   (only needed for --model cagean)
  u2os_bg4/U2OS_bg4_seqs.npy
  u2os_bg4/U2OS_bg4_labels.npy
  u2os_bg4/U2OS_bg4_pqs_ids.txt
  u2os_bg4/U2OS_bg4_{mark}_epi.npy   (only needed for --model cagean)
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
Q_HEK       = 0.03790


def _build_pqs_chr(pqs_bed):
    pqs_chr = {}
    with open(pqs_bed) as fh:
        for line in fh:
            p = line.split("\t", 5)
            pqs_chr[p[3]] = p[0]
    return pqs_chr


def infer(model, seqs, epi, device, batch=512):
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            s = torch.tensor(seqs[i:i+batch], dtype=torch.float32).to(device)
            e = torch.tensor(epi[i:i+batch],  dtype=torch.float32).unsqueeze(1).to(device)
            preds.append(model(torch.cat([s, e], dim=1)).reshape(-1).cpu().numpy())
    return np.concatenate(preds)


def infer_seqonly(model, seqs, device, batch=512):
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            s = torch.tensor(seqs[i:i+batch], dtype=torch.float32).to(device)
            preds.append(model(s).reshape(-1).cpu().numpy())
    return np.concatenate(preds)


def _eval_hek(model, crosscell_dir, mark, pqs_chr, q, device, batch):
    hek_dir    = os.path.join(crosscell_dir, "hek293t")
    labels_all = np.load(os.path.join(hek_dir, "HEK293T_labels.npy"))
    pqs_ids    = open(os.path.join(hek_dir, "HEK293T_pqs_ids.txt")).read().split()
    test_set   = set(TEST_CHROMS)
    test_mask  = np.array([pqs_chr.get(p, "") in test_set for p in pqs_ids])
    tnz_mask   = test_mask & (labels_all > 0)
    lb_tnz     = (labels_all[tnz_mask] > q).astype(np.int32)
    print(f"HEK293T T+NZ: {tnz_mask.sum():,}  pos={lb_tnz.mean()*100:.1f}%")

    seqs_tnz = np.load(os.path.join(hek_dir, "HEK293T_seqs.npy"))[tnz_mask]
    if mark:
        epi_full = np.load(os.path.join(hek_dir, f"HEK293T_{mark}_epi.npy"))
        epi_tnz  = epi_full[tnz_mask].copy(); del epi_full
        preds = infer(model, seqs_tnz, epi_tnz, device, batch)
    else:
        preds = infer_seqonly(model, seqs_tnz, device, batch)
    auprc = average_precision_score(lb_tnz, preds)
    print(f"HEK293T cross-cell AUPRC = {auprc:.4f}")
    return auprc


def _eval_k562(model, crosscell_dir, mark, pqs_chr, device, batch):
    k562_dir   = os.path.join(crosscell_dir, "k562_bg4")
    labels_all = np.load(os.path.join(k562_dir, "K562_bg4_labels.npy"), mmap_mode="r")
    pqs_ids    = open(os.path.join(k562_dir, "K562_bg4_pqs_ids.txt")).read().split()
    test_set   = set(TEST_CHROMS)
    test_mask  = np.array([pqs_chr.get(p, "") in test_set for p in pqs_ids])
    tnz_mask   = test_mask & (labels_all > 0)
    q90        = float(np.percentile(labels_all[tnz_mask], 90))
    lb_tnz     = (labels_all[tnz_mask] > q90).astype(np.int32)
    print(f"K562 BG4 T+NZ: {tnz_mask.sum():,}  Q90={q90:.5f}  pos={lb_tnz.mean()*100:.1f}%")

    seqs_tnz = np.load(os.path.join(k562_dir, "K562_bg4_seqs.npy"), mmap_mode="r")[tnz_mask].copy()
    if mark:
        epi_full = np.load(os.path.join(k562_dir, f"K562_bg4_{mark}_epi.npy"), mmap_mode="r")
        epi_tnz  = epi_full[tnz_mask].copy(); del epi_full
        preds = infer(model, seqs_tnz, epi_tnz, device, batch)
    else:
        preds = infer_seqonly(model, seqs_tnz, device, batch)
    auprc = average_precision_score(lb_tnz, preds)
    print(f"K562 BG4 cross-technique AUPRC = {auprc:.4f}")
    return auprc


def _eval_u2os(model, crosscell_dir, mark, pqs_chr, device, batch):
    """U2OS BG4: binary narrowPeak labels, no T+NZ filter.

    pqs_chr must be built from the genome-wide PQS_padded.bed (not a
    cell-type-specific file) so that all U2OS PQS IDs resolve to a chromosome.
    """
    u2os_dir   = os.path.join(crosscell_dir, "u2os_bg4")
    labels_all = np.load(os.path.join(u2os_dir, "U2OS_bg4_labels.npy"))
    with open(os.path.join(u2os_dir, "U2OS_bg4_pqs_ids.txt")) as fh:
        pqs_ids = fh.read().split()
    seqs_full  = np.load(os.path.join(u2os_dir, "U2OS_bg4_seqs.npy"), mmap_mode="r")
    if not (len(labels_all) == len(pqs_ids) == len(seqs_full)):
        raise RuntimeError(
            f"U2OS array length mismatch: labels={len(labels_all)}, "
            f"pqs_ids={len(pqs_ids)}, seqs={len(seqs_full)}"
        )
    test_set  = set(TEST_CHROMS)
    test_mask = np.array([pqs_chr.get(p, "") in test_set for p in pqs_ids])
    n_test    = int(test_mask.sum())
    if n_test == 0:
        raise RuntimeError("No U2OS PQS IDs mapped to test chromosomes — check pqs_chr source.")
    lb_test   = labels_all[test_mask].astype(np.int32)
    n_pos     = int(lb_test.sum())
    if n_pos == 0:
        print("WARNING: no positive U2OS BG4 labels in test chromosomes — AUPRC undefined.")
        return float("nan")
    print(f"U2OS BG4 test: {n_test:,}  pos={n_pos:,} ({n_pos/n_test*100:.1f}%)")

    seqs_test = seqs_full[test_mask].copy()
    if mark:
        epi_full = np.load(os.path.join(u2os_dir, f"U2OS_bg4_{mark}_epi.npy"), mmap_mode="r")
        epi_test = epi_full[test_mask].copy(); del epi_full
        preds = infer(model, seqs_test, epi_test, device, batch)
    else:
        preds = infer_seqonly(model, seqs_test, device, batch)
    auprc = average_precision_score(lb_test, preds)
    print(f"U2OS BG4 cross-cell+technique AUPRC = {auprc:.4f}")
    return auprc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          required=True, help="Model checkpoint (.pt)")
    p.add_argument("--crosscell_dir", required=True, help="Base directory for crosscell data")
    p.add_argument("--pqs_bed",       required=True, help="PQS_padded.bed file")
    p.add_argument("--model",         default="cagean", choices=["cagean", "seqonly"])
    p.add_argument("--mark",          default=None, choices=["h3k4me3", "h3k27ac", "atac"],
                   help="Required for --model cagean; ignored for seqonly")
    p.add_argument("--q_hek",         default=Q_HEK, type=float)
    p.add_argument("--batch",         default=512,   type=int)
    args = p.parse_args()

    if args.model == "cagean" and args.mark is None:
        p.error("--mark is required when --model cagean")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "seqonly":
        model = SeqOnlyTransformer()
    else:
        model = CAGEAN()
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval().to(device)
    print(f"{args.model}  mark={args.mark or 'seq-only'}  device={device}")

    pqs_chr = _build_pqs_chr(args.pqs_bed)
    _eval_hek( model, args.crosscell_dir, args.mark, pqs_chr, args.q_hek, device, args.batch)
    _eval_k562(model, args.crosscell_dir, args.mark, pqs_chr,              device, args.batch)
    _eval_u2os(model, args.crosscell_dir, args.mark, pqs_chr,              device, args.batch)



if __name__ == "__main__":
    main()
