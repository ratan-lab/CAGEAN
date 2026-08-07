"""
Evaluate the concatenation-fusion ablation (ConcatFusionInstanceNorm) alongside
CAGEAN on four datasets, reproducing Supplementary Table S12.

Datasets evaluated:
  A549 same-cell        — test chromosomes chr1,3,5,7,9 (T+NZ Q90)
  HEK293T cross-cell    — T+NZ protocol (labels > 0 in test chroms)
  K562 BG4 cross-tech.  — T+NZ protocol; Q90 threshold from filtered set
  H1975 cross-cell      — T+NZ protocol; H3K27ac and ATAC only (no H3K4me3)
                          pass --h1975_dir to enable; omitted if not provided

Usage:
  python eval/eval_ablation.py \\
      --ckpt_cagean_h3k4me3   checkpoints/cagean_h3k4me3.pt \\
      --ckpt_cagean_h3k27ac   checkpoints/cagean_h3k27ac.pt \\
      --ckpt_cagean_atac      checkpoints/cagean_atac.pt \\
      --ckpt_ablation_h3k4me3 checkpoints/concat_fusion_h3k4me3.pt \\
      --ckpt_ablation_h3k27ac checkpoints/concat_fusion_h3k27ac.pt \\
      --ckpt_ablation_atac    checkpoints/concat_fusion_atac.pt \\
      --a549_dir              data/a549/test \\
      --crosscell_dir         data/crosscell \\
      --pqs_bed               pqs/PQS_padded.bed \\
      --h1975_dir             data/crosscell/h1975  # optional

Outputs a table in Table S12 format to stdout.
"""
import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cagean import CAGEAN
from models.concat_fusion import ConcatFusionInstanceNorm

TEST_CHROMS = ["chr1", "chr3", "chr5", "chr7", "chr9"]
Q_A549      = 4.08277148e-02
Q_HEK       = 0.03790
Q_H1975     = 0.07343   # T+NZ Q90 for H1975 test chromosomes
MARKS       = ["h3k4me3", "h3k27ac", "atac"]


def _infer(model, seqs, epi, device, batch=512):
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            s = torch.tensor(seqs[i:i+batch], dtype=torch.float32).to(device)
            e = torch.tensor(epi[i:i+batch],  dtype=torch.float32).unsqueeze(1).to(device)
            preds.append(model(torch.cat([s, e], dim=1)).reshape(-1).cpu().numpy())
    return np.concatenate(preds)


def _build_pqs_chr(pqs_bed):
    pqs_chr = {}
    with open(pqs_bed) as fh:
        for line in fh:
            p = line.split("\t", 5)
            pqs_chr[p[3]] = p[0]
    return pqs_chr


def _eval_a549(models_by_mark, a549_dir, device, batch):
    """A549 same-cell: T+NZ Q90 (consistent with bootstrap protocol)."""
    seqs_all, epi_all, labs_all = {}, {}, None
    for mark in MARKS:
        seqs_all[mark] = np.concatenate([
            np.load(os.path.join(a549_dir, f"{c}_seqs.npy")) for c in TEST_CHROMS
        ])
        epi_all[mark] = np.concatenate([
            np.load(os.path.join(a549_dir, f"{c}_epi_{mark}.npy")) for c in TEST_CHROMS
        ])
    labs_raw = np.concatenate([
        np.load(os.path.join(a549_dir, f"{c}_labels.npy")) for c in TEST_CHROMS
    ])
    tnz      = labs_raw > 0
    q90      = float(np.percentile(labs_raw[tnz], 90))
    labels   = (labs_raw[tnz] > q90).astype(np.int32)
    n, pos   = tnz.sum(), labels.mean() * 100
    print(f"A549 same-cell (T+NZ): {n:,}  Q90={q90:.5f}  pos={pos:.1f}%")

    rows = []
    for mark in MARKS:
        seqs = seqs_all[mark][tnz]
        epi  = epi_all[mark][tnz]
        for label, model in models_by_mark[mark].items():
            preds = _infer(model, seqs, epi, device, batch)
            rows.append(("A549 same-cell", mark, label,
                         average_precision_score(labels, preds)))
    return rows


def _eval_crosscell(models_by_mark, data_dir, cell_label, file_prefix,
                    q, pqs_chr, device, batch, skip_marks=None):
    """Generic T+NZ cross-cell evaluation."""
    labels_all = np.load(os.path.join(data_dir, f"{file_prefix}_labels.npy"))
    pqs_ids    = open(os.path.join(data_dir, f"{file_prefix}_pqs_ids.txt")).read().split()
    test_set   = set(TEST_CHROMS)
    test_mask  = np.array([pqs_chr.get(p, "") in test_set for p in pqs_ids])
    tnz_mask   = test_mask & (labels_all > 0)
    labels     = (labels_all[tnz_mask] > q).astype(np.int32)
    n, pos     = tnz_mask.sum(), labels.mean() * 100
    print(f"{cell_label} T+NZ: {n:,}  pos={pos:.1f}%")

    seqs_tnz = np.load(os.path.join(data_dir, f"{file_prefix}_seqs.npy"))[tnz_mask]
    rows = []
    for mark in MARKS:
        if skip_marks and mark in skip_marks:
            continue
        epi_full = np.load(os.path.join(data_dir, f"{file_prefix}_{mark}_epi.npy"))
        epi_tnz  = epi_full[tnz_mask].copy(); del epi_full
        for label, model in models_by_mark[mark].items():
            preds = _infer(model, seqs_tnz, epi_tnz, device, batch)
            rows.append((cell_label, mark, label,
                         average_precision_score(labels, preds)))
    return rows


def _eval_k562(models_by_mark, crosscell_dir, pqs_chr, device, batch):
    """K562 BG4 cross-technique: Q90 computed from filtered set."""
    k562_dir   = os.path.join(crosscell_dir, "k562_bg4")
    labels_all = np.load(os.path.join(k562_dir, "K562_bg4_labels.npy"), mmap_mode="r")
    pqs_ids    = open(os.path.join(k562_dir, "K562_bg4_pqs_ids.txt")).read().split()
    test_set   = set(TEST_CHROMS)
    test_mask  = np.array([pqs_chr.get(p, "") in test_set for p in pqs_ids])
    tnz_mask   = test_mask & (labels_all > 0)
    q90        = float(np.percentile(labels_all[tnz_mask], 90))
    labels     = (labels_all[tnz_mask] > q90).astype(np.int32)
    n, pos     = tnz_mask.sum(), labels.mean() * 100
    print(f"K562 BG4 cross-technique T+NZ: {n:,}  Q90={q90:.5f}  pos={pos:.1f}%")

    seqs_tnz = np.load(os.path.join(k562_dir, "K562_bg4_seqs.npy"),
                       mmap_mode="r")[tnz_mask].copy()
    rows = []
    for mark in MARKS:
        epi_full = np.load(
            os.path.join(k562_dir, f"K562_bg4_{mark}_epi.npy"), mmap_mode="r"
        )
        epi_tnz  = epi_full[tnz_mask].copy(); del epi_full
        for label, model in models_by_mark[mark].items():
            preds = _infer(model, seqs_tnz, epi_tnz, device, batch)
            rows.append(("K562 BG4 cross-technique", mark, label,
                         average_precision_score(labels, preds)))
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Evaluate CAGEAN vs ConcatFusionInstanceNorm (Table S12)."
    )
    for mark in MARKS:
        p.add_argument(f"--ckpt_cagean_{mark}",   required=True)
        p.add_argument(f"--ckpt_ablation_{mark}",  required=True)
    p.add_argument("--a549_dir",      required=True)
    p.add_argument("--crosscell_dir", required=True)
    p.add_argument("--pqs_bed",       required=True)
    p.add_argument("--h1975_dir",     default=None,
                   help="Optional: path to H1975 crosscell arrays (H3K27ac and ATAC only)")
    p.add_argument("--q_hek",    default=Q_HEK,    type=float)
    p.add_argument("--q_h1975",  default=Q_H1975,  type=float)
    p.add_argument("--batch",    default=512,       type=int)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    models_by_mark = {}
    for mark in MARKS:
        cagean_ckpt   = getattr(args, f"ckpt_cagean_{mark}")
        ablation_ckpt = getattr(args, f"ckpt_ablation_{mark}")

        m_cagean = CAGEAN()
        m_cagean.load_state_dict(
            torch.load(cagean_ckpt, map_location=device, weights_only=True)
        )
        m_cagean.eval().to(device)

        m_ablation = ConcatFusionInstanceNorm()
        m_ablation.load_state_dict(
            torch.load(ablation_ckpt, map_location=device, weights_only=True)
        )
        m_ablation.eval().to(device)

        models_by_mark[mark] = {
            "CAGEAN":                    m_cagean,
            "Concat+Res+InstanceNorm":   m_ablation,
        }

    pqs_chr = _build_pqs_chr(args.pqs_bed)
    all_rows = []

    all_rows += _eval_a549(models_by_mark, args.a549_dir, device, args.batch)
    all_rows += _eval_crosscell(
        models_by_mark,
        os.path.join(args.crosscell_dir, "hek293t"), "HEK293T cross-cell",
        "HEK293T", args.q_hek, pqs_chr, device, args.batch
    )
    all_rows += _eval_k562(models_by_mark, args.crosscell_dir, pqs_chr, device, args.batch)

    if args.h1975_dir:
        all_rows += _eval_crosscell(
            models_by_mark,
            args.h1975_dir, "H1975 cross-cell", "H1975",
            args.q_h1975, pqs_chr, device, args.batch,
            skip_marks={"h3k4me3"},   # H3K4me3 data not available for H1975
        )

    # Print Table S12
    print()
    print(f"{'Dataset':<30} {'Mark':<10} {'CAGEAN':>8} {'Concat+Res+InstanceNorm':>24} {'Delta':>7}")
    print("-" * 82)
    dataset_order = ["A549 same-cell", "HEK293T cross-cell",
                     "H1975 cross-cell", "K562 BG4 cross-technique"]
    from collections import defaultdict
    table = defaultdict(dict)
    for dataset, mark, model_label, auprc in all_rows:
        table[(dataset, mark)][model_label] = auprc

    for ds in dataset_order:
        for mark in MARKS:
            key = (ds, mark)
            if key not in table:
                continue
            row  = table[key]
            cag  = row.get("CAGEAN", float("nan"))
            abl  = row.get("Concat+Res+InstanceNorm", float("nan"))
            dlt  = abl - cag
            print(f"{ds:<30} {mark:<10} {cag:8.4f} {abl:24.4f} {dlt:+7.4f}")


if __name__ == "__main__":
    main()
