"""
Constitutive vs. strict CTS stratification of HEK293T T+NZ (Supp Table S10).

Splits the HEK293T T+NZ set by A549 G4P signal at the same PQS locus:
  - Constitutive: HEK293T signal > 0 AND A549 signal > 0 (G4 shared with training cell)
  - Strict CTS:   HEK293T signal > 0 AND A549 signal = 0 (G4 absent in training cell)

Shows that CAGEAN's cross-cell epigenomic advantage is concentrated in constitutive G4s.
At strict CTS sites (A549 signal = 0), all models converge near the 1% positive-rate
floor and epigenomic input provides no gain over seq-only prediction.

Usage:
  python eval/eval_strict_cts.py \\
      --ckpt_h3k4me3  checkpoints/cagean_h3k4me3.pt \\
      --ckpt_h3k27ac  checkpoints/cagean_h3k27ac.pt \\
      --ckpt_atac     checkpoints/cagean_atac.pt \\
      --ckpt_seqonly  checkpoints/seqonly_transformer.pt \\
      --a549_dir      data/a549/test \\
      --crosscell_dir data/crosscell/hek293t \\
      --pqs_bed       pqs/PQS_padded.bed
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


def infer_cagean(model, seqs, epi, device, batch=512):
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            s = torch.tensor(seqs[i:i+batch], dtype=torch.float32).to(device)
            e = torch.tensor(epi[i:i+batch],  dtype=torch.float32).unsqueeze(1).to(device)
            preds.append(model(torch.cat([s, e], dim=1)).squeeze(-1).cpu().numpy())
    return np.concatenate(preds)


def infer_seqonly(model, seqs, device, batch=512):
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            s = torch.tensor(seqs[i:i+batch], dtype=torch.float32).to(device)
            preds.append(model(s).squeeze(-1).cpu().numpy())
    return np.concatenate(preds)


def _build_pqs_chr(pqs_bed):
    pqs_chr = {}
    with open(pqs_bed) as fh:
        for line in fh:
            p = line.strip().split("\t", 5)
            pqs_chr[p[3]] = p[0]
    return pqs_chr


def _build_a549_label_lookup(a549_dir):
    """Build dict {pqs_id: a549_raw_label} from per-chromosome arrays."""
    lookup = {}
    for chrn in TEST_CHROMS:
        ids_path    = os.path.join(a549_dir, f"{chrn}_pqs_ids.txt")
        labels_path = os.path.join(a549_dir, f"{chrn}_labels.npy")
        ids    = open(ids_path).read().split()
        labels = np.load(labels_path)
        for pid, lab in zip(ids, labels):
            lookup[pid] = float(lab)
    return lookup


def _report(name, lb, preds):
    n   = len(lb)
    pos = lb.sum()
    if pos < 10 or (n - pos) < 10:
        print(f"  [{name:20s}]  n={n:7,}  pos={pos:5,} — too few to report")
        return float("nan")
    auprc = average_precision_score(lb, preds)
    print(f"  [{name:20s}]  n={n:7,}  pos={pos:5,} ({100*pos/n:4.1f}%)  AUPRC={auprc:.4f}")
    return auprc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_h3k4me3",  required=True)
    p.add_argument("--ckpt_h3k27ac",  required=True)
    p.add_argument("--ckpt_atac",     required=True)
    p.add_argument("--ckpt_seqonly",  required=True)
    p.add_argument("--a549_dir",      required=True, help="A549 test chromosome arrays")
    p.add_argument("--crosscell_dir", required=True, help="HEK293T arrays directory")
    p.add_argument("--pqs_bed",       required=True)
    p.add_argument("--batch",         default=512, type=int)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Building A549 label lookup...")
    a549_lookup = _build_a549_label_lookup(args.a549_dir)
    pqs_chr     = _build_pqs_chr(args.pqs_bed)
    print(f"  A549 PQS sites (test chroms): {len(a549_lookup):,}")

    print("Loading HEK293T arrays...")
    labels_all = np.load(os.path.join(args.crosscell_dir, "HEK293T_labels.npy"))
    pqs_ids    = open(os.path.join(args.crosscell_dir, "HEK293T_pqs_ids.txt")).read().split()
    seqs_all   = np.load(os.path.join(args.crosscell_dir, "HEK293T_seqs.npy"))

    # T+NZ mask: test chromosomes + HEK293T signal > 0
    test_set  = set(TEST_CHROMS)
    test_mask = np.array([pqs_chr.get(p, "") in test_set for p in pqs_ids])
    tnz_mask  = test_mask & (labels_all > 0)
    print(f"  T+NZ sites: {tnz_mask.sum():,}")

    # A549 signal for T+NZ sites; -1.0 sentinel for any ID not found in lookup
    pqs_tnz  = [p for p, m in zip(pqs_ids, tnz_mask) if m]
    a549_sig = np.array([a549_lookup.get(p, -1.0) for p in pqs_tnz])
    n_missing = (a549_sig < 0).sum()
    # T+NZ is already restricted to test chromosomes, so every ID must be in the A549 lookup
    assert n_missing == 0, (
        f"{n_missing} T+NZ PQS IDs missing from A549 lookup — "
        "all T+NZ sites should be on test chromosomes covered by A549 data"
    )

    # Use a small epsilon for zero-check: labels are stored as float but exact zero
    # at zero-coverage sites is expected; any positive signal, however small, is > 1e-9
    constitutive_mask = a549_sig > 1e-9
    strict_cts_mask   = a549_sig <= 1e-9

    print(f"  Constitutive (A549 > 0): {constitutive_mask.sum():,}")
    print(f"  Strict CTS   (A549 = 0): {strict_cts_mask.sum():,}")
    assert constitutive_mask.sum() + strict_cts_mask.sum() == len(a549_sig)
    print()

    lb_tnz   = (labels_all[tnz_mask] > Q_HEK).astype(np.int32)
    seqs_tnz = seqs_all[tnz_mask]

    strata = [
        ("ALL T+NZ",     np.ones(len(lb_tnz), dtype=bool)),
        ("Constitutive", constitutive_mask),
        ("Strict CTS",   strict_cts_mask),
    ]

    marks = ["h3k4me3", "h3k27ac", "atac"]
    ckpts = {
        "h3k4me3": args.ckpt_h3k4me3,
        "h3k27ac": args.ckpt_h3k27ac,
        "atac":    args.ckpt_atac,
    }

    print("=== SeqOnlyTransformer ===")
    model = SeqOnlyTransformer()
    model.load_state_dict(torch.load(args.ckpt_seqonly, map_location=device, weights_only=True))
    model.eval().to(device)
    preds_seqonly = infer_seqonly(model, seqs_tnz, device, args.batch)
    for name, mask in strata:
        _report(name, lb_tnz[mask], preds_seqonly[mask])
    del model
    print()

    for mark in marks:
        print(f"=== CAGEAN {mark} ===")
        model = CAGEAN()
        model.load_state_dict(torch.load(ckpts[mark], map_location=device, weights_only=True))
        model.eval().to(device)
        epi_tnz = np.load(
            os.path.join(args.crosscell_dir, f"HEK293T_{mark}_epi.npy"))[tnz_mask]
        preds = infer_cagean(model, seqs_tnz, epi_tnz, device, args.batch)
        for name, mask in strata:
            _report(name, lb_tnz[mask], preds[mask])
        del model, epi_tnz
        print()

    print("Done.")


if __name__ == "__main__":
    main()
