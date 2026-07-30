"""
Train seq-only Transformer and CAGEAN on K562 BG4 same-cell data.

Usage — seq-only Transformer (no epigenomic input):
  python train/train_bg4.py --model seq_only \\
      --data_dir data/k562_bg4 --q 0.01842 \\
      --out checkpoints/bg4_seqonly_transformer.pt

Usage — CAGEAN with one epigenomic mark:
  python train/train_bg4.py --model cagean --mark h3k4me3 \\
      --data_dir data/k562_bg4 --q 0.01842 \\
      --out checkpoints/bg4_cagean_h3k4me3.pt

Data directory layout:
  {data_dir}/{chrom}_seqs.npy       (N, 4, 1000) int8 one-hot
  {data_dir}/{chrom}_labels.npy     (N,) float32 BG4 signal
  {data_dir}/{chrom}_epi_{mark}.npy (N, 1000) float32  [CAGEAN only]

The label threshold q is computed as the Q90 of non-zero K562 BG4 labels
within test chromosomes (T+NZ protocol). The default value 0.01842 was used
in the paper. Re-run data_prep/03_make_labels.py to recompute.
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
from models.seqonly_transformer import SeqOnlyTransformer

TRAIN_CHROMS = [
    "chr2", "chr4", "chr6", "chr8", "chr11", "chr12", "chr13",
    "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
    "chr20", "chr21", "chr22", "chrX", "chrY",
]
VAL_CHROM = "chr10"
_criterion = nn.BCEWithLogitsLoss(reduction="none")


class _SeqDataset(Dataset):
    def __init__(self, seqs, labels_bin, weights):
        self.seqs = torch.from_numpy(seqs)
        self.y    = torch.from_numpy(labels_bin)
        self.w    = torch.from_numpy(weights)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.seqs[idx].float(), self.y[idx], self.w[idx]


class _EpiDataset(Dataset):
    def __init__(self, seqs, labels_bin, epi, weights):
        self.seqs = torch.from_numpy(seqs)
        self.epi  = torch.from_numpy(epi)
        self.y    = torch.from_numpy(labels_bin)
        self.w    = torch.from_numpy(weights)

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


def _load_chroms(data_dir, chroms, mark=None):
    all_seqs, all_epi, all_labels = [], [], []
    for chrn in chroms:
        sf = os.path.join(data_dir, f"{chrn}_seqs.npy")
        lf = os.path.join(data_dir, f"{chrn}_labels.npy")
        if not os.path.exists(sf):
            print(f"  WARNING: {chrn} not found, skipping")
            continue
        all_seqs.append(np.load(sf))
        all_labels.append(np.load(lf))
        if mark is not None:
            all_epi.append(np.load(os.path.join(data_dir, f"{chrn}_epi_{mark}.npy")))
    seqs   = np.concatenate(all_seqs)
    labels = np.concatenate(all_labels)
    epi    = np.concatenate(all_epi) if all_epi else None
    return seqs, labels, epi


def _run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    total_loss, total_n = 0.0, 0
    ctx = torch.enable_grad if train else torch.no_grad
    with ctx():
        for batch in loader:
            x, y, w = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            logits = model(x).squeeze(1)
            loss   = (_criterion(logits, y) * w).mean()
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y)
            total_n    += len(y)
    return total_loss / total_n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          required=True, choices=["seq_only", "cagean"])
    p.add_argument("--data_dir",       required=True)
    p.add_argument("--out",            required=True, help="Output checkpoint path (.pt)")
    p.add_argument("--mark",           default=None,  choices=["h3k4me3", "h3k27ac", "atac"],
                   help="Epigenomic mark (required for --model cagean)")
    p.add_argument("--q",              default=0.01842, type=float,
                   help="Label threshold (Q90 of non-zero K562 BG4 T+NZ labels)")
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

    if args.model == "cagean" and args.mark is None:
        p.error("--mark is required for --model cagean")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mark   = args.mark

    if args.model == "seq_only":
        model = SeqOnlyTransformer(
            d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward, stem_kernel=args.stem_kernel,
            dropout=args.dropout,
        ).to(device)
        label = "SeqOnlyTransformer (K562 BG4)"
    else:
        model = CAGEAN(
            d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward, stem_kernel=args.stem_kernel,
            dropout=args.dropout,
        ).to(device)
        label = f"CAGEAN+{mark} (K562 BG4)"

    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"{label}  device={device}  params={n_params:,}  q={args.q}")

    train_chroms = [c for c in TRAIN_CHROMS if c != VAL_CHROM]
    print("Loading training data...")
    seqs_tr, labels_tr, epi_tr = _load_chroms(args.data_dir, train_chroms, mark)
    bin_tr  = (labels_tr > args.q).astype(np.float32)
    n_pos   = int(bin_tr.sum()); n_neg = len(bin_tr) - n_pos
    W_POS   = (n_pos + n_neg) / (2.0 * n_pos)
    W_NEG   = (n_pos + n_neg) / (2.0 * n_neg)
    weights_tr = np.where(bin_tr > 0.5, W_POS, W_NEG).astype(np.float32)
    print(f"  Train: {len(seqs_tr):,}  pos={n_pos:,} ({n_pos/len(seqs_tr)*100:.2f}%)"
          f"  W_POS={W_POS:.1f}  W_NEG={W_NEG:.4f}")

    DS = _EpiDataset if args.model == "cagean" else _SeqDataset
    ds_args = (seqs_tr, bin_tr, epi_tr, weights_tr) if args.model == "cagean" \
              else (seqs_tr, bin_tr, weights_tr)
    train_ds     = DS(*ds_args)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    del seqs_tr, labels_tr, epi_tr, bin_tr, weights_tr

    print(f"Loading val {VAL_CHROM}...")
    seqs_v, labels_v, epi_v = _load_chroms(args.data_dir, [VAL_CHROM], mark)
    bin_v   = (labels_v > args.q).astype(np.float32)
    vn_pos  = int(bin_v.sum()); vn_neg = len(bin_v) - vn_pos
    vW_POS  = len(bin_v) / (2.0 * vn_pos) if vn_pos > 0 else 1.0
    vW_NEG  = len(bin_v) / (2.0 * vn_neg) if vn_neg > 0 else 1.0
    weights_v = np.where(bin_v > 0.5, vW_POS, vW_NEG).astype(np.float32)
    ds_args_v = (seqs_v, bin_v, epi_v, weights_v) if args.model == "cagean" \
                else (seqs_v, bin_v, weights_v)
    val_ds     = DS(*ds_args_v)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"  Val: {len(val_ds):,}  pos={vn_pos/len(val_ds)*100:.2f}%")

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
