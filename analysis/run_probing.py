"""
Layerwise linear probing of SeqOnlyTransformer.

Extracts token-level representations at the CNN stem and each of the four
Transformer layers, then fits linear probes for interpretable sequence
properties. Writes two CSV files consumed by figures/figS3.R:

  figS3_probing_r2.csv    token-level R² (or AUROC for in_pqs) per layer/feature
  figS3_probing_auprc.csv mean-pooled class AUPRC per layer

All probes are evaluated on a held-out 35% test split; no probe touches
any ISM sequences.

Usage (from repo root):
  python analysis/run_probing.py \\
      --ckpt         checkpoints/seqonly_transformer.pt \\
      --a549_test_dir data/a549/test \\
      --pqs_bed      pqs/PQS_unpadded.bed \\
      --pad_bed      pqs/PQS_padded.bed \\
      --out_dir      figures/data
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    p.add_argument("--pqs_bed",       required=True,
                   help="PQS_unpadded.bed (unpadded PQS coordinates)")
    p.add_argument("--pad_bed",       required=True,
                   help="PQS_padded_1kb.bed (1 kb window coordinates)")
    p.add_argument("--out_dir",       default="figures/data",
                   help="Output directory for CSVs (default: figures/data)")
    p.add_argument("--n_probe",  type=int, default=1500,
                   help="Windows to probe (default: 1500)")
    p.add_argument("--chrom",    default="chr1")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--batch",    type=int, default=512,
                   help="Batch size for representation extraction")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Q_A549  = 4.08277148e-02
L       = 1000
STEM_K  = 10
N_TOK   = L // STEM_K
BASES   = ["A", "T", "C", "G"]
B2I     = {b: i for i, b in enumerate(BASES)}
G_IDX   = B2I["G"]
C_IDX   = B2I["C"]

LAYER_KEYS   = ["cnn", "layer1", "layer2", "layer3", "layer4"]
LAYER_LABELS = ["CNN", "Layer 1", "Layer 2", "Layer 3", "Layer 4"]

FEATURE_LABELS = {
    "G_frac":    "G fraction",
    "C_frac":    "C fraction",
    "GC_frac":   "GC fraction",
    "dist_pqs":  "dist(PQS)",
    "in_pqs":    "in_pqs",
    "max_G_run": "max G run",
    "n_G3":      "n G3 runs",
}


# ---------------------------------------------------------------------------
# Model
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


# ---------------------------------------------------------------------------
# Representation extraction (forward hooks)
# ---------------------------------------------------------------------------

class _Tap:
    """Context manager: captures CNN stem and per-layer Transformer outputs."""

    def __init__(self, model):
        self.model    = model
        self.acts     = {}
        self._handles = []

    def __enter__(self):
        def _save(name):
            def hook(mod, inp, out):
                self.acts[name] = out.detach().float().cpu()
            return hook

        # seq_stem output is (B, d_model, N_TOK) — permuted in extract_reps
        self._handles.append(
            self.model.seq_stem.register_forward_hook(_save("cnn"))
        )
        for li, layer in enumerate(self.model.seq_encoder.layers):
            self._handles.append(layer.register_forward_hook(_save(f"layer{li+1}")))
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def extract_reps(model, seqs, device, batch=512):
    """Returns dict: layer_key → (N, N_TOK, d_model) float32 numpy array."""
    import torch

    store = defaultdict(list)
    n_batches = (len(seqs) + batch - 1) // batch
    with torch.no_grad():
        for bi, i in enumerate(range(0, len(seqs), batch)):
            xb = torch.from_numpy(np.asarray(seqs[i:i+batch])).float().to(device)
            with _Tap(model) as tap:
                model(xb)
                store["cnn"].append(tap.acts["cnn"].permute(0, 2, 1).contiguous().numpy())
                for li in range(4):
                    store[f"layer{li+1}"].append(tap.acts[f"layer{li+1}"].numpy())
            print(f"  batch {bi+1}/{n_batches}", flush=True)

    return {k: np.concatenate(v) for k, v in store.items()}


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
    assert seqs.ndim == 3 and seqs.shape[1] == 4 and seqs.shape[2] == L, \
        f"Expected seqs shape (N, 4, {L}), got {seqs.shape}"
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
# Token-level feature computation
# ---------------------------------------------------------------------------

def compute_token_features(indices, data):
    """
    Returns dict: feature_name → (N, N_TOK) float32 array.
    Each token covers STEM_K=10 bp (non-overlapping, matching Conv1d stride).
    dist_pqs: signed distance (bp) from token midpoint to PQS midpoint.
    in_pqs: 1 if token overlaps the annotated PQS, 0 otherwise.
    """
    n = len(indices)
    T = {k: np.zeros((n, N_TOK), np.float32) for k in FEATURE_LABELS}

    for r, i in enumerate(indices):
        oh  = data["seqs"][i]
        g   = oh[G_IDX] > 0
        c   = oh[C_IDX] > 0
        s   = data["pqs_start"][i]
        e   = data["pqs_end"][i]
        mid = (s + e) / 2.0

        for t in range(N_TOK):
            a, b = t * STEM_K, (t + 1) * STEM_K
            gg = g[a:b]; cc = c[a:b]
            T["G_frac"][r, t]   = gg.mean()
            T["C_frac"][r, t]   = cc.mean()
            T["GC_frac"][r, t]  = (gg | cc).mean()
            T["dist_pqs"][r, t] = (a + b) / 2.0 - mid
            T["in_pqs"][r, t]   = float(a < e and b > s)

            if gg.any():
                d    = np.diff(np.concatenate(([False], gg, [False])).astype(np.int8))
                runs = np.where(d == -1)[0] - np.where(d == 1)[0]
                T["max_G_run"][r, t] = runs.max()
                T["n_G3"][r, t]      = (runs >= 3).sum()

    return T


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def probe_token_level(reps, token_feats, tr_idx, te_idx):
    """
    Fit a linear probe per layer per feature on token-level embeddings.
    Ridge regression → R²; logistic regression (balanced) → AUROC for in_pqs.
    Train/test split is at window level so all tokens from a window stay together.
    """
    rows = []
    for lkey, llabel in zip(LAYER_KEYS, LAYER_LABELS):
        X     = reps[lkey]   # (N, N_TOK, d)
        X_tr  = X[tr_idx].reshape(-1, X.shape[-1])
        X_te  = X[te_idx].reshape(-1, X.shape[-1])
        sc    = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr)
        X_te_s = sc.transform(X_te)

        for feat_key, feat_label in FEATURE_LABELS.items():
            y    = token_feats[feat_key]
            y_tr = y[tr_idx].ravel()
            y_te = y[te_idx].ravel()

            if feat_key == "in_pqs":
                m      = LogisticRegression(max_iter=1000, random_state=0,
                                            class_weight="balanced").fit(X_tr_s, y_tr)
                score  = roc_auc_score(y_te, m.decision_function(X_te_s))
                metric = "AUROC"
            else:
                m      = Ridge(alpha=1.0).fit(X_tr_s, y_tr)
                score  = r2_score(y_te, m.predict(X_te_s))
                metric = "R2"

            rows.append(dict(layer=llabel, feature=feat_label,
                             score=float(score), metric=metric))
        print(f"  probed {llabel}", flush=True)
    return rows


def probe_pooled_class(reps, labels, tr_idx, te_idx):
    """
    Mean-pool token representations (matching model head), fit logistic probe,
    report AUPRC. Uses class_weight='balanced' for the imbalanced G4 label.
    """
    rows  = []
    y_tr  = labels[tr_idx]
    y_te  = labels[te_idx]
    pos_rate = float(labels.mean())

    for lkey, llabel in zip(LAYER_KEYS, LAYER_LABELS):
        P   = reps[lkey].mean(axis=1)   # (N, d) — same pooling as model head
        sc  = StandardScaler().fit(P[tr_idx])
        m   = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=0).fit(sc.transform(P[tr_idx]), y_tr)
        dec = m.decision_function(sc.transform(P[te_idx]))
        rows.append(dict(layer=llabel,
                         auprc=float(average_precision_score(y_te, dec)),
                         auroc=float(roc_auc_score(y_te, dec)),
                         pos_rate=pos_rate))
    return rows


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

    # Load chromosome
    print(f"loading {args.chrom} ...", flush=True)
    pqs_df = load_pqs_bed(pqs_bed, pad_bed)
    data   = load_chrom(args.chrom, test_dir, pqs_df)
    y      = data["label"]
    print(f"{args.chrom}: {len(y):,} windows  pos={y.mean()*100:.1f}%", flush=True)

    # Select probe set (PQS fully inside window, pqs_len ≤ 300)
    ok     = ((data["pqs_start"] >= 0) & (data["pqs_end"] <= L) &
              ((data["pqs_end"] - data["pqs_start"]) <= 300))
    ok_idx = np.where(ok)[0]
    assert len(ok_idx) > 0, f"No windows passed the filter on {args.chrom}"
    sel    = rng.choice(ok_idx, min(args.n_probe, len(ok_idx)), replace=False)
    print(f"probing {len(sel)} windows  ({y[sel].mean()*100:.1f}% positive)",
          flush=True)

    # Extract representations
    print("extracting representations ...", flush=True)
    t0   = time.time()
    reps = extract_reps(model, data["seqs"][sel], device, args.batch)
    print(f"done in {time.time()-t0:.0f}s", flush=True)

    # Compute token-level features
    print("computing token features ...", flush=True)
    token_feats = compute_token_features(sel, data)

    # Stratified train/test split at window level
    tr_idx, te_idx = train_test_split(
        np.arange(len(sel)), test_size=0.35, random_state=args.seed,
        stratify=y[sel]
    )

    # Token-level probes → figS3_probing_r2.csv
    print("fitting token-level probes ...", flush=True)
    token_rows = probe_token_level(reps, token_feats, tr_idx, te_idx)
    token_df   = pd.DataFrame(token_rows)
    token_df[["layer", "feature", "score", "metric"]].to_csv(
        out_dir / "figS3_probing_r2.csv", index=False
    )

    pivot = (token_df[token_df["feature"].isin(["dist(PQS)", "G fraction", "GC fraction"])]
             .pivot(index="feature", columns="layer", values="score")
             [LAYER_LABELS])
    print("\nToken-level probe scores (R²):")
    print(pivot.round(3).to_string(), flush=True)

    # Pooled class probes → figS3_probing_auprc.csv
    print("\nfitting pooled class probes ...", flush=True)
    pool_rows = probe_pooled_class(reps, y[sel], tr_idx, te_idx)
    pool_df   = pd.DataFrame(pool_rows)
    pool_df.to_csv(out_dir / "figS3_probing_auprc.csv", index=False)

    print("\nPooled class AUPRC:")
    for _, row in pool_df.iterrows():
        print(f"  {row['layer']:8s}  AUPRC={row['auprc']:.4f}  AUROC={row['auroc']:.4f}",
              flush=True)

    print(f"\noutputs written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
