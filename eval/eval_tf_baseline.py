"""
Evaluate the retrained epiG4NN TensorFlow baseline (Tables 1 and 2).

Requires the TF environment:
  conda activate cagean_tf
  python eval/eval_tf_baseline.py --ckpt_dir checkpoints/tf_epig4nn_h3k4me3 ...

epiG4NN (Korsak et al. 2022) was retrained from scratch using the original
TensorFlow implementation on the same A549 data, chromosome split, and labels
as CAGEAN. Using the original TF implementation — rather than a PyTorch
reimplementation — avoids attributing any performance difference to
reimplementation artifacts.

TF environment: environment_tf.yml
  Requires TensorFlow 2.x and the epiG4NN repository cloned separately.
  See: https://github.com/anyakors/epiG4NN

This script assumes epiG4NN model weights are in SavedModel or checkpoint
format as produced by the original training code. Adapt _load_tf_model() to
match the exact checkpoint format of your retrained epiG4NN models.
"""
import os
import sys
import argparse
import json
import numpy as np
from sklearn.metrics import average_precision_score

TEST_CHROMS = ["chr1", "chr3", "chr5", "chr7", "chr9"]
Q_A549      = 4.08277148e-02
Q_HEK       = 0.03790


def _load_tf_model(ckpt_dir):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow not found. Activate the TF environment:\n"
            "  conda activate cagean_tf\n"
            "  conda env create -f environment_tf.yml"
        )
    return tf.saved_model.load(ckpt_dir)


def _infer_tf(model, seqs, epi, batch=256):
    import tensorflow as tf
    preds = []
    for i in range(0, len(seqs), batch):
        s = tf.constant(seqs[i:i+batch], dtype=tf.float32)   # (B, 4, 1000)
        e = tf.constant(epi[i:i+batch,  None, :], dtype=tf.float32)  # (B, 1, 1000)
        x = tf.concat([s, e], axis=1)   # (B, 5, 1000); adjust if model uses NHWC
        out = model(x, training=False)
        preds.append(out.numpy().ravel())
    return np.concatenate(preds)


def _infer_tf_seqonly(model, seqs, batch=256):
    import tensorflow as tf
    preds = []
    for i in range(0, len(seqs), batch):
        s   = tf.constant(seqs[i:i+batch], dtype=tf.float32)
        out = model(s, training=False)
        preds.append(out.numpy().ravel())
    return np.concatenate(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mark",           required=True, choices=["h3k4me3", "h3k27ac", "atac"])
    p.add_argument("--ckpt_dir",       required=True,
                   help="SavedModel directory for epiG4NN retrained with --mark")
    p.add_argument("--ckpt_seqonly",   default=None,
                   help="SavedModel directory for sequence-only epiG4NN retrain (optional)")
    p.add_argument("--a549_test_dir",  required=True)
    p.add_argument("--crosscell_dir",  required=True)
    p.add_argument("--pqs_bed",        required=True)
    p.add_argument("--norm_params",    default=None,
                   help="JSON file with z-norm {mean, std} from 04_normalize.py; "
                        "apply only if TF models were retrained on z-normalized input")
    p.add_argument("--batch",          default=256, type=int)
    args = p.parse_args()

    norm_mean = norm_std = None
    if args.norm_params:
        with open(args.norm_params) as fh:
            _p = json.load(fh)
        norm_mean, norm_std = _p["mean"], _p["std"]
        print(f"Z-norm params loaded: mean={norm_mean:.6f}  std={norm_std:.6f}")

    print(f"Loading epiG4NN TF model from {args.ckpt_dir}")
    model = _load_tf_model(args.ckpt_dir)

    # A549 same-cell
    all_seqs, all_epi, all_labels = [], [], []
    for chrn in TEST_CHROMS:
        all_seqs.append(  np.load(os.path.join(args.a549_test_dir, f"{chrn}_seqs.npy")))
        all_labels.append(np.load(os.path.join(args.a549_test_dir, f"{chrn}_labels.npy")))
        all_epi.append(   np.load(os.path.join(args.a549_test_dir, f"{chrn}_epi_{args.mark}.npy")))
    seqs       = np.concatenate(all_seqs)
    epi        = np.concatenate(all_epi)
    if norm_mean is not None:
        epi = (epi - norm_mean) / (norm_std + 1e-8)
    labels_bin = (np.concatenate(all_labels) > Q_A549).astype(np.int32)
    preds      = _infer_tf(model, seqs, epi, args.batch)
    auprc_a549 = average_precision_score(labels_bin, preds)
    print(f"A549 same-cell AUPRC = {auprc_a549:.4f}  (mark={args.mark})")

    # Sequence-only epiG4NN (Δ(arch) baseline for Table 2)
    if args.ckpt_seqonly:
        model_seq = _load_tf_model(args.ckpt_seqonly)
        preds_seq = _infer_tf_seqonly(model_seq, seqs, args.batch)
        auprc_seq = average_precision_score(labels_bin, preds_seq)
        print(f"A549 seq-only  AUPRC = {auprc_seq:.4f}  Δ={auprc_a549-auprc_seq:+.4f}")

    # HEK293T cross-cell — build pqs_chr then T+NZ mask
    pqs_chr = {}
    with open(args.pqs_bed) as fh:
        for line in fh:
            parts = line.split("\t", 5)
            pqs_chr[parts[3]] = parts[0]
    hek_dir    = args.crosscell_dir
    labels_all = np.load(os.path.join(hek_dir, "HEK293T_labels.npy"))
    pqs_ids    = open(os.path.join(hek_dir, "HEK293T_pqs_ids.txt")).read().split()
    test_set   = set(TEST_CHROMS)
    test_mask  = np.array([pqs_chr.get(p, "") in test_set for p in pqs_ids])
    tnz_mask   = test_mask & (labels_all > 0)
    lb_tnz     = (labels_all[tnz_mask] > Q_HEK).astype(np.int32)
    seqs_tnz   = np.load(os.path.join(hek_dir, "HEK293T_seqs.npy"))[tnz_mask]
    epi_full   = np.load(os.path.join(hek_dir, f"HEK293T_{args.mark}_epi.npy"))
    epi_tnz    = epi_full[tnz_mask].copy(); del epi_full
    if norm_mean is not None:
        epi_tnz = (epi_tnz - norm_mean) / (norm_std + 1e-8)
    preds_hek  = _infer_tf(model, seqs_tnz, epi_tnz, args.batch)
    auprc_hek  = average_precision_score(lb_tnz, preds_hek)
    print(f"HEK293T cross-cell AUPRC = {auprc_hek:.4f}  (mark={args.mark})")


if __name__ == "__main__":
    main()
