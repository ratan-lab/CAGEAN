"""
Per-base saturation mutagenesis (ISM) for SeqOnlyTransformer.

Runs ISM on N_ISM positives + N_ISM negatives from chr1 of the A549 test set
and writes four output files:
  ism_pos.npz / ism_neg.npz   raw (N,4,1000) delta-logit arrays + indices
  ism_in_out.csv              inside/outside PQS importance per sequence
  ism_substitution_stats.csv  mean/median delta grouped by region × ref × alt
  ism_summary.json            headline ratios and p-values for downstream use

Usage (from repo root):
  python analysis/run_ism.py \\
      --ckpt          checkpoints/seqonly_transformer.pt \\
      --a549_test_dir data/a549/test \\
      --pqs_bed       pqs/PQS_unpadded.bed \\
      --pad_bed       pqs/PQS_padded.bed \\
      --out_dir       figures/data \\
      --n_ism         1000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.seqonly_transformer import SeqOnlyTransformer


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt",          required=True,
                   help="SeqOnlyTransformer checkpoint (.pt)")
    p.add_argument("--a549_test_dir", required=True,
                   help="A549 test directory: {chrom}_seqs.npy, _labels.npy, _pqs_ids.txt")
    p.add_argument("--pqs_bed",  required=True,
                   help="PQS_unpadded.bed (unpadded PQS coordinates)")
    p.add_argument("--pad_bed",  required=True,
                   help="PQS_padded_1kb.bed (1 kb window coordinates)")
    p.add_argument("--out_dir",  required=True,
                   help="Output directory (created if absent)")
    p.add_argument("--n_ism",    type=int, default=1000,
                   help="Sequences per class (default: 1000)")
    p.add_argument("--chrom",    default="chr1",
                   help="Chromosome to sample from (default: chr1)")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--batch",    type=int, default=3000,
                   help="Forward-pass batch size for ISM mutations")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Constants (must match training data prep)
# ---------------------------------------------------------------------------

Q_A549  = 4.08277148e-02   # top-5% G4P binarisation threshold
L       = 1000             # window length
STEM_K  = 10               # conv stride → 100 tokens
BASES   = ["A", "T", "C", "G"]   # one-hot channel order in the npy files
B2I     = {b: i for i, b in enumerate(BASES)}
G_IDX   = B2I["G"]


# ---------------------------------------------------------------------------
# Model + inference
# ---------------------------------------------------------------------------

def load_model(ckpt: Path, device):
    import torch
    model = SeqOnlyTransformer(d_model=64, nhead=4, num_layers=4,
                               dim_feedforward=192, stem_kernel=STEM_K, dropout=0.0)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded {ckpt.name}  |  {n:,} params  |  device={device}", flush=True)
    return model


def predict_logits(model, x, device, batch=2048):
    import torch
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x))
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i+batch].float().to(device, non_blocking=True)
            out.append(model(xb).squeeze(1).float().cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pqs_bed(pqs_bed: Path, pad_bed: Path):
    unp = pd.read_csv(pqs_bed, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    pad = pd.read_csv(pad_bed, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    assert len(unp) == len(pad) and (unp["name"].values == pad["name"].values).all(), \
        "unpadded/padded BEDs are not row-matched"
    unp["pid"]       = unp["name"].str.split("_").str[0]
    assert unp["pid"].is_unique, "duplicate PIDs after name.split('_')[0]"
    unp["win_start"] = pad["start"].values
    unp["win_end"]   = pad["end"].values
    return unp.set_index("pid")


def load_chrom(chrom: str, test_dir: Path, pqs_bed_df: pd.DataFrame):
    seqs   = np.load(test_dir / f"{chrom}_seqs.npy")
    signal = np.load(test_dir / f"{chrom}_labels.npy")
    with open(test_dir / f"{chrom}_pqs_ids.txt") as fh:
        ids = [l.strip() for l in fh]
    assert len(ids) == len(seqs) == len(signal)

    pid = pd.Series(ids).str.split("_").str[0].values
    sub = pqs_bed_df.reindex(pid)
    assert sub["chrom"].notna().all(), f"{chrom}: unmatched PQS ids"

    pqs_s = (sub["start"].values - sub["win_start"].values).astype(int)
    pqs_e = (sub["end"].values   - sub["win_start"].values).astype(int)

    return dict(seqs=seqs, signal=signal,
                label=(signal > Q_A549).astype(np.int8),
                pqs_start=pqs_s, pqs_end=pqs_e)


# ---------------------------------------------------------------------------
# ISM
# ---------------------------------------------------------------------------

def ism_one(model, oh, device, batch):
    """oh: (4,L) int8 → (4,L) delta-logit; ref base's own row is 0."""
    import torch
    ref = torch.from_numpy(np.ascontiguousarray(oh)).float()

    ref_idx = oh.argmax(0)
    muts, coord = [], []
    for pos in range(L):
        for b in range(4):
            if b == ref_idx[pos]:
                continue
            m = ref.clone(); m[:, pos] = 0.0; m[b, pos] = 1.0
            muts.append(m); coord.append((b, pos))
    muts  = torch.stack(muts)
    coord = np.array(coord)  # (3000, 2) for vectorised scatter

    out = np.zeros((4, L), np.float32)
    with torch.no_grad():
        base_logit = model(ref[None].to(device)).item()
        for i in range(0, len(muts), batch):
            lg = model(muts[i:i+batch].to(device)).squeeze(1).cpu().numpy()
            bp = coord[i:i+batch]
            out[bp[:, 0], bp[:, 1]] = lg - base_logit
    return out, base_logit


def run_ism(model, indices, data, device, batch, cache_path: Path, tag: str):
    if cache_path.exists():
        z = np.load(cache_path)
        print(f"[{tag}] loaded cache  shape={z['delta'].shape}", flush=True)
        return z["delta"], z["base"], z["idx"]

    deltas, bases = [], []
    t0 = time.time()
    for k, i in enumerate(indices):
        d, bl = ism_one(model, data["seqs"][i], device, batch)
        deltas.append(d); bases.append(bl)
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            eta = el / (k + 1) * (len(indices) - k - 1)
            print(f"[{tag}] {k+1}/{len(indices)}  {el:.0f}s  ETA {eta:.0f}s", flush=True)

    deltas = np.stack(deltas)
    bases  = np.array(bases)
    np.savez_compressed(cache_path, delta=deltas, base=bases, idx=indices)
    print(f"[{tag}] saved {cache_path}  ({time.time()-t0:.0f}s total)", flush=True)
    return deltas, bases, np.asarray(indices)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def per_pos_importance(delta):
    """(N,4,L) → (N,L) mean |Δ| over the 3 non-reference substitutions."""
    return np.abs(delta).sum(1) / 3.0


def in_out_split(imp, indices, data):
    ins, outs = [], []
    for r, i in enumerate(indices):
        s, e = data["pqs_start"][i], data["pqs_end"][i]
        m = np.zeros(L, bool); m[max(0, s):min(L, e)] = True
        if m.sum() < 5 or (~m).sum() < 5:
            continue
        ins.append(imp[r, m].mean()); outs.append(imp[r, ~m].mean())
    return np.array(ins), np.array(outs)


def build_substitution_df(delta, indices, data):
    recs = []
    for r, i in enumerate(indices):
        oh  = data["seqs"][i]
        ref = oh.argmax(0)
        s, e = max(0, data["pqs_start"][i]), min(L, data["pqs_end"][i])
        reg  = np.where((np.arange(L) >= s) & (np.arange(L) < e),
                        "inside PQS", "flank")
        for b in range(4):
            mask = ref != b
            recs.append(pd.DataFrame({
                "ref":    np.array(BASES)[ref[mask]],
                "alt":    BASES[b],
                "region": reg[mask],
                "d":      delta[r, b, mask],
            }))
    return pd.concat(recs, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import torch

    args     = parse_args()
    rng      = np.random.default_rng(args.seed)
    ckpt     = Path(args.ckpt)
    test_dir = Path(args.a549_test_dir)
    pqs_bed  = Path(args.pqs_bed)
    pad_bed  = Path(args.pad_bed)
    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for p in [ckpt, test_dir, pqs_bed, pad_bed]:
        assert p.exists(), f"missing: {p}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}  |  device: {device}", flush=True)

    model = load_model(ckpt, device)

    # AUPRC sanity check on chr1
    print("verifying checkpoint against published AUPRC ...", flush=True)
    seqs_c1   = np.load(test_dir / "chr1_seqs.npy", mmap_mode="r")
    signal_c1 = np.load(test_dir / "chr1_labels.npy")
    y_c1      = (signal_c1 > Q_A549).astype(np.int8)
    from sklearn.metrics import average_precision_score
    logits_c1 = predict_logits(model, seqs_c1, device)
    auprc     = average_precision_score(y_c1, logits_c1)
    print(f"chr1 AUPRC = {auprc:.4f}  (expect ~0.726)", flush=True)
    assert abs(auprc - 0.7257) < 0.05, \
        f"AUPRC {auprc:.4f} is far from 0.7257 — investigate before using ISM results"

    # Load full chromosome for ISM
    print(f"loading {args.chrom} ...", flush=True)
    pqs_df = load_pqs_bed(pqs_bed, pad_bed)
    data   = load_chrom(args.chrom, test_dir, pqs_df)
    y      = data["label"]
    print(f"{args.chrom}: {len(y):,} windows  pos={y.mean()*100:.1f}%", flush=True)

    # Select indices (PQS fully inside window, pqs_len ≤ 300)
    ok      = ((data["pqs_start"] >= 0) & (data["pqs_end"] <= L) &
               ((data["pqs_end"] - data["pqs_start"]) <= 300))
    pos_idx = np.where((y == 1) & ok)[0]
    neg_idx = np.where((y == 0) & ok)[0]
    sel_pos = rng.choice(pos_idx, min(args.n_ism, len(pos_idx)), replace=False)
    sel_neg = rng.choice(neg_idx, min(args.n_ism, len(neg_idx)), replace=False)
    print(f"ISM: {len(sel_pos)} pos + {len(sel_neg)} neg  "
          f"= {(len(sel_pos)+len(sel_neg))*L*3:,} forward passes", flush=True)

    # Run ISM (with caching)
    ism_pos, base_pos, idx_pos = run_ism(
        model, sel_pos, data, device, args.batch,
        out_dir / "ism_pos.npz", "pos")
    ism_neg, base_neg, idx_neg = run_ism(
        model, sel_neg, data, device, args.batch,
        out_dir / "ism_neg.npz", "neg")

    # Inside / outside PQS
    print("computing inside/outside importance ...", flush=True)
    imp_pos = per_pos_importance(ism_pos)
    imp_neg = per_pos_importance(ism_neg)

    pi, po = in_out_split(imp_pos, idx_pos, data)
    ni, no = in_out_split(imp_neg, idx_neg, data)

    in_out_df = pd.concat([
        pd.DataFrame({"class": "pos", "region": "inside",  "imp": pi}),
        pd.DataFrame({"class": "pos", "region": "outside", "imp": po}),
        pd.DataFrame({"class": "neg", "region": "inside",  "imp": ni}),
        pd.DataFrame({"class": "neg", "region": "outside", "imp": no}),
    ], ignore_index=True)
    in_out_df.to_csv(out_dir / "ism_in_out.csv", index=False)

    wx_pos = stats.wilcoxon(pi, po, alternative="greater")
    wx_neg = stats.wilcoxon(ni, no, alternative="greater")
    ratio_pos = float(np.median(pi) / np.median(po))
    ratio_neg = float(np.median(ni) / np.median(no))
    print(f"positives inside/outside ratio = {ratio_pos:.2f}×  p={wx_pos.pvalue:.2e}", flush=True)
    print(f"negatives inside/outside ratio = {ratio_neg:.2f}×  p={wx_neg.pvalue:.2e}", flush=True)

    # Substitution stats (positives only — the key causal claim)
    print("building substitution table ...", flush=True)
    S_pos = build_substitution_df(ism_pos, idx_pos, data)
    sub_stats = (S_pos.groupby(["region", "ref", "alt"])["d"]
                 .agg(["mean", "median", "std", "count"])
                 .reset_index())
    sub_stats.to_csv(out_dir / "ism_substitution_stats.csv", index=False)

    # Directional G-integrity summary
    inside = S_pos[S_pos.region == "inside PQS"]
    loseG  = inside[inside.ref == "G"]["d"]
    loseO  = inside[inside.ref != "G"]["d"]
    gainG  = inside[(inside.alt == "G") & (inside.ref != "G")]["d"]
    mwu    = stats.mannwhitneyu(loseG, loseO, alternative="less")
    print(f"inside PQS: lose G mean Δ = {loseG.mean():+.4f}  "
          f"lose A/T/C mean Δ = {loseO.mean():+.4f}  "
          f"gain G mean Δ = {gainG.mean():+.4f}  "
          f"Mann-Whitney p = {mwu.pvalue:.2e}", flush=True)

    # Summary JSON
    summary = {
        "n_ism_per_class": int(args.n_ism),
        "chrom":           args.chrom,
        "seed":            args.seed,
        "auprc_chr1":      float(auprc),
        "pos_in_out_ratio": ratio_pos,
        "neg_in_out_ratio": ratio_neg,
        "wilcoxon_pos_p":  float(wx_pos.pvalue),
        "wilcoxon_neg_p":  float(wx_neg.pvalue),
        "inside_loseG_mean_delta":  float(loseG.mean()),
        "inside_loseO_mean_delta":  float(loseO.mean()),
        "inside_gainG_mean_delta":  float(gainG.mean()),
        "mannwhitney_g_vs_nonG_p":  float(mwu.pvalue),
    }
    with open(out_dir / "ism_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nall outputs written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
