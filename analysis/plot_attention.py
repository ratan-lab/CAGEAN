"""
Convert attention numpy arrays to CSVs for figures/fig4.R.

Reads outputs of extract_attention.py from --results_dir and writes four CSV
files to --out_dir (default: figures/data/).

Output files:
  fig4a.csv   mark, group, position_bp, epi_marg
  fig4b.csv   group, seq_bin, epi_bin, attn_weight
  fig4c.csv   group, position_bp, row_entropy
  fig4c_ggg.csv  position_bp, ggg_density

Usage:
  python analysis/plot_attention.py \\
      --results_dir analysis/results \\
      --out_dir     figures/data
"""
import os
import argparse
import numpy as np
import pandas as pd

MARKS = ["h3k4me3", "h3k27ac", "atac"]

# Groups shown in panel (a): marginal attention profiles
PANEL_A_GROUPS = [
    "a549_pos_all", "a549_pos_promoter",
    "hek_pos_promoter", "hek_pos_enhancer", "a549_neg",
]

# Groups shown in panel (b): heatmaps (H3K27ac only)
PANEL_B_MARK   = "h3k27ac"
PANEL_B_GROUPS = ["a549_pos_promoter", "a549_pos_enhancer", "a549_neg"]

# Groups shown in panel (c): row entropy + GGG (H3K27ac G4-positive)
PANEL_C_MARK   = "h3k27ac"
PANEL_C_GROUPS = [
    "a549_pos_all", "a549_pos_promoter", "a549_pos_enhancer",
    "hek_pos_promoter", "hek_pos_enhancer",
]

# Map internal group names to display labels
GROUP_LABELS = {
    "a549_pos_all":      "A549 G4+ all",
    "a549_pos_promoter": "A549 G4+ promoter",
    "a549_pos_enhancer": "A549 G4+ enhancer",
    "a549_neg":          "A549 G4-",
    "hek_pos_promoter":  "HEK293T G4+ promoter",
    "hek_pos_enhancer":  "HEK293T G4+ enhancer",
}

MARK_LABELS = {
    "h3k4me3": "H3K4me3",
    "h3k27ac": "H3K27ac",
    "atac":    "ATAC",
}

BIN_CENTERS = np.arange(100) * 10 + 5   # 5, 15, ..., 995 bp


def _load(results_dir, mark, group, suffix):
    path = os.path.join(results_dir, f"{mark}_{group}_{suffix}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="analysis/results")
    p.add_argument("--out_dir",     default="figures/data")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── fig4a: marginal attention profiles ───────────────────────────────────
    rows_a = []
    for mark in MARKS:
        for group in PANEL_A_GROUPS:
            arr = _load(args.results_dir, mark, group, "epi_marg")
            if arr is None:
                print(f"  MISSING: {mark} {group} epi_marg — skipping")
                continue
            for pos, val in zip(BIN_CENTERS, arr):
                rows_a.append({
                    "mark":        MARK_LABELS[mark],
                    "group":       GROUP_LABELS[group],
                    "position_bp": int(pos),
                    "epi_marg":    float(val),
                })
    df_a = pd.DataFrame(rows_a)
    df_a.to_csv(os.path.join(args.out_dir, "fig4a.csv"), index=False)
    print(f"Wrote fig4a.csv  ({len(df_a):,} rows)")

    # ── fig4b: heatmap data (H3K27ac) ────────────────────────────────────────
    rows_b = []
    for group in PANEL_B_GROUPS:
        mat = _load(args.results_dir, PANEL_B_MARK, group, "attn_matrix")
        if mat is None:
            print(f"  MISSING: {PANEL_B_MARK} {group} attn_matrix — skipping")
            continue
        for seq_bin in range(100):
            for epi_bin in range(100):
                rows_b.append({
                    "group":      GROUP_LABELS[group],
                    "seq_bin":    seq_bin,
                    "epi_bin":    epi_bin,
                    "attn_weight": float(mat[seq_bin, epi_bin]),
                })
    df_b = pd.DataFrame(rows_b)
    df_b.to_csv(os.path.join(args.out_dir, "fig4b.csv"), index=False)
    print(f"Wrote fig4b.csv  ({len(df_b):,} rows)")

    # ── fig4c: row entropy ────────────────────────────────────────────────────
    rows_c = []
    for group in PANEL_C_GROUPS:
        arr = _load(args.results_dir, PANEL_C_MARK, group, "row_entropy")
        if arr is None:
            print(f"  MISSING: {PANEL_C_MARK} {group} row_entropy — skipping")
            continue
        for pos, val in zip(BIN_CENTERS, arr):
            rows_c.append({
                "group":       GROUP_LABELS[group],
                "position_bp": int(pos),
                "row_entropy": float(val),
            })
    df_c = pd.DataFrame(rows_c)
    df_c.to_csv(os.path.join(args.out_dir, "fig4c.csv"), index=False)
    print(f"Wrote fig4c.csv  ({len(df_c):,} rows)")

    # ── fig4c_ggg: GGG density (averaged across groups, H3K27ac pos) ─────────
    ggg_arrays = []
    for group in PANEL_C_GROUPS:
        arr = _load(args.results_dir, PANEL_C_MARK, group, "ggg")
        if arr is not None:
            ggg_arrays.append(arr)
    if ggg_arrays:
        ggg_mean = np.stack(ggg_arrays).mean(axis=0)
        df_ggg = pd.DataFrame({
            "position_bp": BIN_CENTERS.astype(int),
            "ggg_density": ggg_mean.astype(float),
        })
        df_ggg.to_csv(os.path.join(args.out_dir, "fig4c_ggg.csv"), index=False)
        print(f"Wrote fig4c_ggg.csv")
    else:
        print("  WARNING: no GGG arrays found for fig4c_ggg.csv")

    print("Done. Run: Rscript figures/fig4.R")


if __name__ == "__main__":
    main()
