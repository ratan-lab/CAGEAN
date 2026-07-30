"""
Evaluate CAGEAN or SeqOnlyTransformer on A549 same-cell test chromosomes (chr1,3,5,7,9).

Usage:
  # CAGEAN
  python eval/eval_samecell.py \\
      --ckpt     checkpoints/cagean_h3k4me3.pt \\
      --data_dir data/a549/test \\
      --mark     h3k4me3

  # Seq-only Transformer (no epi input; --mark not required)
  python eval/eval_samecell.py \\
      --model    seqonly \\
      --ckpt     checkpoints/seqonly_transformer.pt \\
      --data_dir data/a549/test

Outputs AUPRC to stdout.
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
Q_A549      = 4.08277148e-02


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     required=True, help="Model checkpoint (.pt)")
    p.add_argument("--data_dir", required=True, help="Directory with test chromosome arrays")
    p.add_argument("--model",    default="cagean", choices=["cagean", "seqonly"])
    p.add_argument("--mark",     default=None,  choices=["h3k4me3", "h3k27ac", "atac"],
                   help="Required for --model cagean; ignored for seqonly")
    p.add_argument("--q",        default=Q_A549, type=float, help="G4 occupancy threshold")
    p.add_argument("--batch",    default=512,    type=int)
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
    n = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    label = f"{args.model}  mark={args.mark or 'seq-only'}  params={n:,}  device={device}"
    print(label)

    all_seqs, all_epi, all_labels = [], [], []
    for chrn in TEST_CHROMS:
        all_seqs.append(  np.load(os.path.join(args.data_dir, f"{chrn}_seqs.npy")))
        all_labels.append(np.load(os.path.join(args.data_dir, f"{chrn}_labels.npy")))
        if args.mark:
            all_epi.append(np.load(os.path.join(args.data_dir, f"{chrn}_epi_{args.mark}.npy")))

    seqs       = np.concatenate(all_seqs)
    labels_bin = (np.concatenate(all_labels) > args.q).astype(np.int32)
    print(f"A549 same-cell: {len(seqs):,} sites  pos={labels_bin.mean()*100:.1f}%")

    if args.model == "seqonly":
        preds = infer_seqonly(model, seqs, device, args.batch)
    else:
        epi   = np.concatenate(all_epi)
        preds = infer(model, seqs, epi, device, args.batch)

    auprc = average_precision_score(labels_bin, preds)
    print(f"AUPRC = {auprc:.4f}")


if __name__ == "__main__":
    main()
