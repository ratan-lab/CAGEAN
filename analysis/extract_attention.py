"""
Extract cross-attention weights from CAGEAN and aggregate by genomic group.

Groups:
  a549_pos_all       A549 G4-positive sites (all)
  a549_pos_promoter  A549 G4-positive, within ±2 kb of a TSS
  a549_pos_enhancer  A549 G4-positive, mean H3K27ac > Q75 of non-zero means
  a549_neg           A549 G4-negative sites
  hek_pos_promoter   HEK293T G4-positive, promoter
  hek_pos_enhancer   HEK293T G4-positive, enhancer-like

Outputs saved to --out_dir:
  {mark}_{group}_attn_matrix.npy   (100, 100) group-averaged attention
  {mark}_{group}_epi_marg.npy      (100,)     column-wise mean (epi marginal)
  {mark}_{group}_row_entropy.npy   (100,)     per-row Shannon entropy (nats)
  {mark}_{group}_ggg.npy           (100,)     GGG trinucleotide density per 10 bp bin

Usage:
  python analysis/extract_attention.py \\
      --ckpt_h3k4me3 checkpoints/cagean_h3k4me3.pt \\
      --ckpt_h3k27ac checkpoints/cagean_h3k27ac.pt \\
      --ckpt_atac    checkpoints/cagean_atac.pt \\
      --a549_test_dir   data/a549/test \\
      --crosscell_dir   data/crosscell/hek293t \\
      --pqs_bed         pqs/PQS_padded.bed \\
      --tss_bed         data/tss_2kb_hg19.bed \\
      --out_dir         analysis/results
"""
import os
import sys
import argparse
import numpy as np
import torch
import pybedtools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cagean import CAGEAN

TEST_CHROMS = ["chr1", "chr3", "chr5", "chr7", "chr9"]
Q_A549      = 4.08277148e-02
Q_HEK       = 0.03790
BATCH       = 256

GROUPS = [
    "a549_pos_all", "a549_pos_promoter", "a549_pos_enhancer",
    "a549_neg", "hek_pos_promoter", "hek_pos_enhancer",
]


def _load_model(ckpt, device):
    model = CAGEAN()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)


def _hook_batch(model, seqs_b, epi_b, device):
    store = {}
    def _h(module, inp, out):
        if out[1] is not None:
            store["w"] = out[1].detach().cpu()
    handle = model.cross_attn.register_forward_hook(_h)
    s = torch.tensor(seqs_b, dtype=torch.float32).to(device)
    e = torch.tensor(epi_b,  dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        model(torch.cat([s, e], dim=1))
    handle.remove()
    return store.get("w")   # (B, 100, 100) or None


def _ggg_bins(seqs_b):
    G       = seqs_b[:, 2, :].astype(np.float32)
    ggg     = G[:, :-2] * G[:, 1:-1] * G[:, 2:]
    ggg_pad = np.pad(ggg, ((0, 0), (0, 2)))
    return ggg_pad.reshape(len(seqs_b), 100, 10).mean(axis=2)  # (B, 100)


def _row_entropy(mat, eps=1e-12):
    return -(mat * np.log(mat + eps)).sum(axis=1)  # (100,)


def collect(model, seqs, epi, group_masks, device):
    attn_sum = {g: np.zeros((100, 100), np.float64) for g in GROUPS}
    ggg_sum  = {g: np.zeros(100,        np.float64) for g in GROUPS}
    counts   = {g: 0                                 for g in GROUPS}

    for i in range(0, len(seqs), BATCH):
        sb = seqs[i:i+BATCH]
        eb = epi[i:i+BATCH]
        attn = _hook_batch(model, sb, eb, device)
        if attn is None:
            continue
        attn_np = attn.numpy()          # (B, 100, 100)
        ggg_np  = _ggg_bins(sb)         # (B, 100)

        for g in GROUPS:
            bm = group_masks[g][i:i+BATCH]
            if not bm.any():
                continue
            attn_sum[g] += attn_np[bm].sum(axis=0)
            ggg_sum[g]  += ggg_np[bm].sum(axis=0)
            counts[g]   += int(bm.sum())

    results = {}
    for g in GROUPS:
        n = counts[g]
        if n == 0:
            print(f"  WARNING: group {g} has 0 samples")
            continue
        mat  = (attn_sum[g] / n).astype(np.float32)
        ggg  = (ggg_sum[g]  / n).astype(np.float32)
        results[g] = {
            "matrix":      mat,
            "epi_marg":    mat.mean(axis=0),     # column-wise mean
            "row_entropy": _row_entropy(mat),
            "ggg":         ggg,
            "n":           n,
        }
        print(f"  {g}: n={n:,}")
    return results


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_h3k4me3",  required=True)
    p.add_argument("--ckpt_h3k27ac",  required=True)
    p.add_argument("--ckpt_atac",     required=True)
    p.add_argument("--a549_test_dir", required=True)
    p.add_argument("--crosscell_dir", required=True)
    p.add_argument("--pqs_bed",       required=True)
    p.add_argument("--tss_bed",       required=True)
    p.add_argument("--out_dir",       default="analysis/results")
    p.add_argument("--h3k27ac_qtile", default=75, type=float)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts  = {"h3k4me3": args.ckpt_h3k4me3,
              "h3k27ac": args.ckpt_h3k27ac,
              "atac":    args.ckpt_atac}

    print("Building annotation sets...")
    promoter_set = _build_promoter_set(args.pqs_bed, args.tss_bed)
    pqs_chr      = _build_pqs_chr(args.pqs_bed)

    # ── A549 test data ─────────────────────────────────────────────────────────
    print("Loading A549 test data...")
    a549_seqs, a549_labels, a549_epi_by_mark = [], [], {}
    for chrn in TEST_CHROMS:
        a549_seqs.append(  np.load(os.path.join(args.a549_test_dir, f"{chrn}_seqs.npy")))
        a549_labels.append(np.load(os.path.join(args.a549_test_dir, f"{chrn}_labels.npy")))
    a549_seqs   = np.concatenate(a549_seqs)
    a549_labels = np.concatenate(a549_labels)

    for mark in ckpts:
        a549_epi_by_mark[mark] = np.concatenate([
            np.load(os.path.join(args.a549_test_dir, f"{chrn}_epi_{mark}.npy"))
            for chrn in TEST_CHROMS
        ])

    # A549 H3K27ac for enhancer annotation
    epi_mean_a549 = a549_epi_by_mark["h3k27ac"].mean(axis=1)
    epi_nz        = epi_mean_a549[epi_mean_a549 > 0]
    enh_thresh_a549 = float(np.percentile(epi_nz, args.h3k27ac_qtile)) if len(epi_nz) > 0 else 0.0
    print(f"  A549 H3K27ac Q{args.h3k27ac_qtile:.0f} of per-site means: {enh_thresh_a549:.5f}")

    a549_bin = (a549_labels > Q_A549)
    a549_pqs = []
    with open(args.pqs_bed) as fh:
        for line in fh:
            p_ = line.split("\t", 5)
            if pqs_chr.get(p_[3], "") in TEST_CHROMS:
                a549_pqs.append(p_[3])
    # pqs IDs aligned to a549_seqs (same chromosome order)
    pqs_to_idx = {}
    idx = 0
    with open(args.pqs_bed) as fh:
        for line in fh:
            p_ = line.split("\t", 5)
            if pqs_chr.get(p_[3], "") in TEST_CHROMS:
                pqs_to_idx[p_[3]] = idx
                idx += 1

    # Build A549 group masks
    n_a549 = len(a549_seqs)
    masks_a549 = {g: np.zeros(n_a549, bool) for g in GROUPS}
    pqs_list_a549 = []
    with open(args.pqs_bed) as fh:
        for line in fh:
            p_ = line.split("\t", 5); pid = p_[3]
            if pqs_chr.get(pid, "") in TEST_CHROMS:
                pqs_list_a549.append(pid)

    for i, pid in enumerate(pqs_list_a549):
        is_pos  = bool(a549_bin[i])
        is_prom = pid in promoter_set
        is_enh  = epi_mean_a549[i] >= enh_thresh_a549
        if is_pos:
            masks_a549["a549_pos_all"][i] = True
            if is_prom:
                masks_a549["a549_pos_promoter"][i] = True
            elif is_enh:
                masks_a549["a549_pos_enhancer"][i] = True
        else:
            masks_a549["a549_neg"][i] = True

    # ── HEK293T T+NZ data ─────────────────────────────────────────────────────
    print("Loading HEK293T T+NZ data...")
    labels_hek = np.load(os.path.join(args.crosscell_dir, "HEK293T_labels.npy"))
    pqs_ids_hek = open(os.path.join(args.crosscell_dir, "HEK293T_pqs_ids.txt")).read().split()
    test_set   = set(TEST_CHROMS)
    test_mask  = np.array([pqs_chr.get(p, "") in test_set for p in pqs_ids_hek])
    tnz_mask   = test_mask & (labels_hek > 0)
    lb_tnz     = (labels_hek[tnz_mask] > Q_HEK)
    pqs_tnz    = [p for p, m in zip(pqs_ids_hek, tnz_mask) if m]
    seqs_hek   = np.load(os.path.join(args.crosscell_dir, "HEK293T_seqs.npy"))[tnz_mask]

    epi_h3k27ac_hek   = np.load(
        os.path.join(args.crosscell_dir, "HEK293T_h3k27ac_epi.npy"))[tnz_mask]
    epi_mean_hek      = epi_h3k27ac_hek.mean(axis=1)
    epi_nz_hek        = epi_mean_hek[epi_mean_hek > 0]
    enh_thresh_hek    = float(np.percentile(epi_nz_hek, args.h3k27ac_qtile)) \
                        if len(epi_nz_hek) > 0 else 0.0
    print(f"  HEK293T H3K27ac Q{args.h3k27ac_qtile:.0f}: {enh_thresh_hek:.5f}")
    del epi_h3k27ac_hek

    n_hek = len(seqs_hek)
    masks_hek = {g: np.zeros(n_hek, bool) for g in GROUPS}
    for i, (pid, is_pos) in enumerate(zip(pqs_tnz, lb_tnz)):
        if not is_pos:
            continue
        if pid in promoter_set:
            masks_hek["hek_pos_promoter"][i] = True
        elif epi_mean_hek[i] >= enh_thresh_hek:
            masks_hek["hek_pos_enhancer"][i] = True

    # ── Extract per mark ───────────────────────────────────────────────────────
    for mark, ckpt in ckpts.items():
        print(f"\n=== {mark} ===")
        model = _load_model(ckpt, device)

        epi_a549 = a549_epi_by_mark[mark]
        epi_hek  = np.load(
            os.path.join(args.crosscell_dir, f"HEK293T_{mark}_epi.npy"))[tnz_mask]

        # Combine A549 and HEK data; split masks accordingly
        seqs_all = np.concatenate([a549_seqs, seqs_hek])
        epi_all  = np.concatenate([epi_a549,  epi_hek])
        n_a      = len(a549_seqs)

        full_masks = {}
        for g in GROUPS:
            m = np.zeros(len(seqs_all), bool)
            m[:n_a]  = masks_a549.get(g, np.zeros(n_a, bool))
            m[n_a:]  = masks_hek.get( g, np.zeros(n_hek, bool))
            full_masks[g] = m

        print("  Collecting attention weights...")
        res = collect(model, seqs_all, epi_all, full_masks, device)
        del model

        for g, r in res.items():
            prefix = os.path.join(args.out_dir, f"{mark}_{g}")
            np.save(f"{prefix}_attn_matrix.npy",  r["matrix"])
            np.save(f"{prefix}_epi_marg.npy",     r["epi_marg"])
            np.save(f"{prefix}_row_entropy.npy",  r["row_entropy"])
            np.save(f"{prefix}_ggg.npy",          r["ggg"])
            print(f"  Saved {g} (n={r['n']:,})")

    print("\nDone. Run analysis/plot_attention.py to export CSVs for figures/fig4.R")


if __name__ == "__main__":
    main()
