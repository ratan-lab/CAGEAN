"""
Train CAGEAN on one or more G4P cell lines.

Single-cell (A549 only):
  python train/train_cagean.py \\
      --cells a549:data/a549:0.0408 \\
      --mark h3k4me3 \\
      --out checkpoints/cagean_h3k4me3.pt

Multi-cell (A549 + HeLa.S3 + HepG2):
  python train/train_cagean.py \\
      --cells a549:data/a549:0.0408 helas3:data/helas3:auto hepg2:data/hepg2:0.5 \\
      --mark h3k4me3 \\
      --val_cell a549 \\
      --out checkpoints/multicell_cagean_h3k4me3.pt

--cells format:  name:data_dir:q_threshold
  q_threshold = float (G4P Q90 value) or "auto" (computed as Q90 of non-zero
  training labels — use for cell lines lacking a pre-computed threshold)

Data directory layout (one directory per cell line):
  {dir}/{chrom}_seqs.npy          (N, 4, 1000) int8 one-hot
  {dir}/{chrom}_labels.npy        (N,) float32 G4P signal
  {dir}/{chrom}_epi_{mark}.npy    (N, 1000) float32 normalised signal
  {dir}/{chrom}_covered_{mark}.npy  (N,) bool  [optional coverage mask]
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cagean import CAGEAN

TRAIN_CHROMS = [
    "chr2", "chr4", "chr6", "chr8", "chr11", "chr12", "chr13",
    "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
    "chr20", "chr21", "chr22", "chrX", "chrY",
]

_criterion = nn.BCEWithLogitsLoss(reduction="none")


class _G4Dataset(Dataset):
    def __init__(self, seqs, labels_bin, epi, sample_weights):
        self.seqs = torch.from_numpy(seqs)
        self.epi  = torch.from_numpy(epi)
        self.y    = torch.from_numpy(labels_bin)
        self.w    = torch.from_numpy(sample_weights)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq = self.seqs[idx].float()
        epi = self.epi[idx].unsqueeze(0)
        return torch.cat([seq, epi], dim=0), self.y[idx], self.w[idx]


def _make_param_groups(model, weight_decay):
    no_decay = {"bias", "LayerNorm", "BatchNorm", "InstanceNorm"}
    decay, no_dec = [], []
    for name, param in model.named_parameters():
        (no_dec if any(k in name for k in no_decay) else decay).append(param)
    return [
        {"params": decay,  "weight_decay": weight_decay},
        {"params": no_dec, "weight_decay": 0.0},
    ]


def _load_cell(name, cell_dir, q, mark, val_chrom):
    avail_s = {f.split("_")[0] for f in os.listdir(cell_dir) if f.endswith("_seqs.npy")}
    avail_e = {f.split("_")[0] for f in os.listdir(cell_dir)
               if f.endswith(f"_epi_{mark}.npy")}
    chroms  = [c for c in TRAIN_CHROMS if c in avail_s and c in avail_e and c != val_chrom]
    if not chroms:
        raise RuntimeError(f"No valid chromosomes for {name} in {cell_dir}")

    all_seqs, all_epi, all_labels = [], [], []
    for chrn in chroms:
        seqs   = np.load(os.path.join(cell_dir, f"{chrn}_seqs.npy"))
        labels = np.load(os.path.join(cell_dir, f"{chrn}_labels.npy"))
        epi    = np.load(os.path.join(cell_dir, f"{chrn}_epi_{mark}.npy"))
        cp     = os.path.join(cell_dir, f"{chrn}_covered_{mark}.npy")
        if os.path.exists(cp):
            mask = np.load(cp)
            seqs, labels, epi = seqs[mask], labels[mask], epi[mask]
        all_seqs.append(seqs); all_epi.append(epi); all_labels.append(labels)

    seqs_all   = np.concatenate(all_seqs)
    epi_all    = np.concatenate(all_epi)
    labels_all = np.concatenate(all_labels)

    if q is None:
        nz = labels_all[labels_all > 0]
        q  = float(np.percentile(nz, 90)) if len(nz) > 0 else 0.0
        print(f"  {name}: q=auto → Q90={q:.6f}")
    else:
        print(f"  {name}: q={q:.6f}")

    labels_bin = (labels_all > q).astype(np.float32)
    n_pos = int(labels_bin.sum())
    n_neg = len(labels_bin) - n_pos
    W_POS = (n_pos + n_neg) / (2.0 * n_pos) if n_pos > 0 else 1.0
    W_NEG = (n_pos + n_neg) / (2.0 * n_neg) if n_neg > 0 else 1.0
    print(f"  {name}: {len(seqs_all):,} samples  pos={n_pos:,} ({n_pos/len(seqs_all)*100:.1f}%)"
          f"  W_POS={W_POS:.2f}  W_NEG={W_NEG:.4f}")
    weights = np.where(labels_bin > 0.5, W_POS, W_NEG).astype(np.float32)
    return seqs_all, epi_all, labels_bin, weights, q


def _run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    total_loss, total_n = 0.0, 0
    ctx = torch.enable_grad if train else torch.no_grad
    with ctx():
        for x, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            logits = model(x).squeeze(1)
            loss   = (_criterion(logits, y) * w).mean()
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y)
            total_n    += len(y)
    return total_loss / total_n


def _parse_cells(specs):
    configs = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected name:dir:q, got: {spec!r}")
        name, d, q_str = parts
        q = None if q_str.lower() == "auto" else float(q_str)
        configs.append((name, d, q))
    return configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", required=True, nargs="+",
                   help="name:dir:q, e.g. a549:data/a549:0.0408  or  hela:data/hela:auto")
    p.add_argument("--mark",           required=True, choices=["h3k4me3", "h3k27ac", "atac"])
    p.add_argument("--out",            required=True, help="Output checkpoint path (.pt)")
    p.add_argument("--val_cell",       default=None,
                   help="Which cell to use for validation (default: first cell)")
    p.add_argument("--val_chrom",      default="chr10")
    p.add_argument("--d_model",        default=64,   type=int)
    p.add_argument("--nhead",          default=4,    type=int)
    p.add_argument("--num_layers",     default=4,    type=int)
    p.add_argument("--dim_feedforward",default=192,  type=int)
    p.add_argument("--stem_kernel",    default=10,   type=int)
    p.add_argument("--dropout",        default=0.0,  type=float)
    p.add_argument("--batch",          default=256,  type=int)
    p.add_argument("--lr",             default=1e-3, type=float)
    p.add_argument("--min_lr",         default=1e-5, type=float)
    p.add_argument("--weight_decay",   default=0.01, type=float)
    p.add_argument("--epochs",         default=50,   type=int)
    p.add_argument("--patience",       default=7,    type=int)
    p.add_argument("--num_workers",    default=4,    type=int)
    args = p.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = _parse_cells(args.cells)
    val_cell = (args.val_cell or configs[0][0]).lower()

    model = CAGEAN(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, stem_kernel=args.stem_kernel,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CAGEAN  mark={args.mark}  cells={[c[0] for c in configs]}")
    print(f"device={device}  params={n_params:,}")

    print("\nLoading training data...")
    all_seqs, all_epi, all_labels, all_weights = [], [], [], []
    val_dir, val_q = None, None
    for cell_name, cell_dir, q_raw in configs:
        seqs, epi, lbin, weights, q = _load_cell(
            cell_name, cell_dir, q_raw, args.mark, args.val_chrom)
        all_seqs.append(seqs); all_epi.append(epi)
        all_labels.append(lbin); all_weights.append(weights)
        if cell_name.lower() == val_cell:
            val_dir, val_q = cell_dir, q

    if val_dir is None:
        raise RuntimeError(f"--val_cell {val_cell!r} not found in --cells")

    seqs_cat    = np.concatenate(all_seqs)
    epi_cat     = np.concatenate(all_epi)
    labels_cat  = np.concatenate(all_labels)
    weights_cat = np.concatenate(all_weights)
    del all_seqs, all_epi, all_labels, all_weights

    n_pos = int(labels_cat.sum())
    print(f"Total train: {len(seqs_cat):,}  pos={n_pos:,} ({n_pos/len(seqs_cat)*100:.1f}%)")

    train_ds     = _G4Dataset(seqs_cat, labels_cat, epi_cat, weights_cat)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    del seqs_cat, epi_cat, labels_cat, weights_cat

    print(f"\nLoading val {args.val_chrom} from {val_cell}...")
    val_seqs   = np.load(os.path.join(val_dir, f"{args.val_chrom}_seqs.npy"))
    val_labels = np.load(os.path.join(val_dir, f"{args.val_chrom}_labels.npy"))
    val_epi    = np.load(os.path.join(val_dir, f"{args.val_chrom}_epi_{args.mark}.npy"))
    cp         = os.path.join(val_dir, f"{args.val_chrom}_covered_{args.mark}.npy")
    if os.path.exists(cp):
        mask = np.load(cp)
        val_seqs, val_labels, val_epi = val_seqs[mask], val_labels[mask], val_epi[mask]
    val_bin  = (val_labels > val_q).astype(np.float32)
    vn_pos   = int(val_bin.sum()); vn_neg = len(val_bin) - vn_pos
    vW_POS   = len(val_bin) / (2.0 * vn_pos) if vn_pos > 0 else 1.0
    vW_NEG   = len(val_bin) / (2.0 * vn_neg) if vn_neg > 0 else 1.0
    val_w    = np.where(val_bin > 0.5, vW_POS, vW_NEG).astype(np.float32)
    val_ds   = _G4Dataset(val_seqs, val_bin, val_epi, val_w)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"  Val: {len(val_ds):,}  pos={vn_pos/len(val_ds)*100:.1f}%")

    param_groups = _make_param_groups(model, args.weight_decay)
    optimizer    = torch.optim.AdamW(param_groups, lr=args.lr)
    t_max        = max(1, args.epochs // 2)
    scheduler    = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.min_lr)
    print(f"Cosine LR: {args.lr} → {args.min_lr} over {t_max} epochs")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    best_val, no_improve = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        tr = _run_epoch(model, train_loader, optimizer, device, train=True)
        vl = _run_epoch(model, val_loader,   optimizer, device, train=False)
        print(f"epoch {epoch:3d}/{args.epochs}  lr={lr:.2e}  train={tr:.4f}  val={vl:.4f}")
        scheduler.step()
        if vl < best_val:
            best_val, no_improve = vl, 0
            torch.save(model.state_dict(), args.out)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}  best_val={best_val:.4f}")
                break

    print(f"Done. Saved: {args.out}")


if __name__ == "__main__":
    main()
