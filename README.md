# CAGEAN: Cross-Attention G-quadruplex Epigenomics Architecture Network

CAGEAN is a PyTorch deep learning model that predicts G-quadruplex (G4) formation in cells
from DNA sequence and epigenomic signal. It uses a strided-convolutional sequence stem,
an epigenomic stem with instance normalization (enabling cross-cell-type generalization),
and a multi-head cross-attention module that fuses the two streams. A matched
sequence-only Transformer (no epigenomic input) is included to quantify the independent
contributions of architecture design and epigenomic data.

Paper: [citation pending]

---

## Repository layout

```
models/             Core PyTorch architecture files
data_prep/          Scripts 01–04: raw data → model-ready numpy arrays
train/              Training scripts and SLURM submission files
eval/               One evaluation script per paper table
analysis/           Attention extraction and figure data generation
figures/            R scripts and CSV data for each manuscript figure
environment.yml     PyTorch environment (all training and eval)
environment_tf.yml  TensorFlow environment (epiG4NN baseline only)
reproduce_tables.sh Fast reproduction from Zenodo weights
```

---

## Environments

Two conda environments are needed:

```bash
# PyTorch — all training, evaluation, and analysis
conda env create -f environment.yml
conda activate cagean

# TensorFlow — epiG4NN baseline columns in Tables 1–2 only
conda env create -f environment_tf.yml
conda activate cagean_tf
```

All scripts use the `cagean` environment unless the script header says otherwise.
`eval_tf_baseline.py` requires `cagean_tf` and will raise a clear ImportError if run
in the wrong environment.

---

## Fast reproduction (≈1 hour, single GPU)

Download the Zenodo archive (DOI: [pending]) and run:

```bash
conda activate cagean
bash reproduce_tables.sh /path/to/zenodo
```

This evaluates all Zenodo-provided model weights on the preprocessed arrays and prints
the numbers for every table in the paper. The epiG4NN baseline columns (Tables 1–2)
require a separate run in the `cagean_tf` environment; `reproduce_tables.sh` prints the
exact commands to run.

### Zenodo archive layout

```
zenodo/
├── checkpoints/
│   ├── cagean_{h3k4me3,h3k27ac,atac}.pt     # CAGEAN A549 single-cell (exp27)
│   ├── multicell_cagean_{mark}.pt            # CAGEAN multi-cell (exp28)
│   ├── seqonly_transformer.pt                # Seq-only Transformer A549 (exp29)
│   ├── bg4_seqonly_transformer.pt            # K562 BG4 seq-only (exp30)
│   ├── bg4_cagean_{mark}.pt                  # K562 BG4 CAGEAN (exp31a-c)
│   ├── tf_epig4nn_{mark}/                    # epiG4NN TF checkpoints
│   ├── tf_epig4nn_seqonly/                   # epiG4NN TF seq-only (optional; for Δarch row)
│   └── norm_params_{mark}.json               # z-norm stats (epiG4NN only, if applicable)
├── data/
│   ├── a549/{train,test}/                    # {chrom}_seqs.npy, _epi_{mark}.npy, _labels.npy
│   ├── crosscell/
│   │   ├── hek293t/                          # HEK293T_{seqs,labels,pqs_ids}.npy/txt
│   │   │                                     # HEK293T_{h3k4me3,h3k27ac,atac}_epi.npy
│   │   └── k562_bg4/                         # K562_bg4_{seqs,labels,pqs_ids}.npy/txt
│   │                                         # K562_bg4_{mark}_epi.npy
│   ├── h9esc/test/
│   ├── k562_bg4_samecell/
│   └── tss_2kb_hg19.bed
└── pqs/
    └── PQS_padded.bed
```

---

## Full reproduction (days of GPU compute)

### Step 1 — Install dependencies

```bash
conda env create -f environment.yml
conda activate cagean
```

### Step 2 — Download raw data

`data_prep/data_manifest.csv` lists every GEO and ENCODE accession used, with cell type,
mark, and download URL. Download the BigWig and reference files to a local directory
before running the pipeline.

### Step 3 — Scan for G4 motifs (PQS universe)

```bash
python data_prep/01_scan_pqs.py \
    --genome  /path/to/hg19.fa \
    --out     pqs/PQS_padded.bed \
    --score   1.2
```

This scans hg19 with G4Hunter (threshold 1.2), extends each hit to a 1000 bp centered
window, and writes `PQS_padded.bed`. All subsequent scripts reference this file.

### Step 4 — Extract epigenomic signal

```bash
python data_prep/02_extract_epi_signal.py \
    --pqs_bed pqs/PQS_padded.bed \
    --bigwig  /path/to/h3k4me3.bigwig \
    --mark    h3k4me3 \
    --out_dir data/a549
```

Repeat for each mark (`h3k4me3`, `h3k27ac`, `atac`) and each cell type. Writes per-
chromosome `{chrom}_seqs.npy`, `{chrom}_epi_{mark}.npy`, and `{chrom}_covered_{mark}.npy`.

### Step 5 — Make labels

```bash
# G4P signal (A549, HEK293T): mean BigWig signal per window
python data_prep/03_make_labels.py \
    --pqs_bed pqs/PQS_padded.bed \
    --bigwig  /path/to/g4p.bigwig \
    --out_dir data/a549

# BG4 peaks (K562): binary from narrow-peak BED
python data_prep/03_make_labels.py \
    --pqs_bed pqs/PQS_padded.bed \
    --bed     /path/to/k562_bg4_peaks.bed \
    --out_dir data/k562_bg4
```

### Step 6 — (epiG4NN baseline only) Z-normalize signal

CAGEAN uses InstanceNorm1d internally and does **not** require pre-normalized input.
This step is only needed for the epiG4NN TF baseline:

```bash
conda activate cagean_tf
python data_prep/04_normalize.py \
    --a549_train_dir data/a549 \
    --mark           h3k4me3 \
    --params_out     data/norm_params_h3k4me3.json \
    --apply_to       data/hek293t data/k562_bg4
```

### Step 7 — Train

**Single-cell CAGEAN (A549, reproduces Table 1 single-cell results):**

```bash
conda activate cagean
python train/train_cagean.py \
    --cells  a549:data/a549:0.0408 \
    --mark   h3k4me3 \
    --out    checkpoints/cagean_h3k4me3.pt \
    --epochs 30
```

**Multi-cell CAGEAN (pooled A549 + HeLa.S3 + HepG2, reproduces Tables 5–6):**

```bash
python train/train_cagean.py \
    --cells     a549:data/a549:0.0408 helas3:data/helas3:auto hepg2:data/hepg2:0.5 \
    --mark      h3k4me3 \
    --val_cell  a549 \
    --out       checkpoints/multicell_cagean_h3k4me3.pt \
    --epochs    30
```

The `--cells` flag accepts `name:data_dir:q_threshold` triples. `q=auto` sets the
binarization threshold to the Q90 of non-zero training labels for that cell line.

**K562 BG4:**

```bash
python train/train_bg4.py \
    --model seq_only --out checkpoints/bg4_seqonly.pt --epochs 30
python train/train_bg4.py \
    --model cagean --mark h3k4me3 --out checkpoints/bg4_cagean_h3k4me3.pt --epochs 30
```

SLURM submission scripts for each experiment are in `train/slurm/`.

### Step 8 — Evaluate

```bash
bash reproduce_tables.sh /path/to/zenodo  # same Zenodo root as fast reproduction
```

Or run individual eval scripts — each takes a `--ckpt` and `--data_dir`:

```bash
python eval/eval_samecell.py  --ckpt checkpoints/cagean_h3k4me3.pt ...
python eval/eval_crosscell.py --ckpt checkpoints/cagean_h3k4me3.pt ...
python eval/eval_stratified.py ...
python eval/eval_h9esc.py ...
```

### Step 9 — Reproduce figures

After running `analysis/extract_attention.py` to generate numpy attention arrays and
`analysis/plot_attention.py` to convert them to `figures/data/fig4*.csv`:

```r
# In R (figures/ directory):
source("utils.R")
source("fig1.R")
source("fig2.R")
# etc.
```

`utils.R` provides the shared theme (`theme_nar()`), colorblind-safe palettes
(Okabe-Ito), and `save_fig()` for 600 dpi TIFF + PDF output. All data is read from
`figures/data/*.csv`; no external file paths are needed.

---

## Evaluation notes

**T+NZ protocol.** Cross-cell evaluation restricts test chromosomes (chr1, 3, 5, 7, 9)
to sites with non-zero G4-occupancy signal in the evaluation cell type. For HEK293T and
K562, where labels are continuous G4P/BG4 signal tracks, this means sites where any
signal was detected (`labels > 0`). For H9 ESC, where labels are binary BG4 peaks,
`labels > 0` would reduce the evaluation set to G4+ sites only and make AUPRC
trivial; instead, `eval_h9esc.py` uses the epigenomic coverage mask
(`{chrom}_covered_{mark}.npy`) to select sites where the mark was detectable, regardless
of G4 status. Both filters exclude constitutively inactive regions not represented in
the training distribution.

**Enhancer threshold calculation (eval_stratified.py).** The enhancer-like stratum uses
Q75 of *per-site mean* H3K27ac signal among non-zero sites. The correct order is:

```python
epi_mean = epi_h3k27ac.mean(axis=1)   # per-site mean — FIRST
epi_nz   = epi_mean[epi_mean > 0]
thresh   = np.percentile(epi_nz, 75)  # ≈ 0.00121 for HEK293T T+NZ
```

Computing the percentile on the raw 2D array before averaging inflates the threshold
by ~13% and silently misclassifies ~17 k sites into the inactive stratum.

**epiG4NN baseline.** Tables 1–2 include results from the original epiG4NN [CITATION]
TF implementation retrained on our A549 data. We deliberately used the original TF
codebase to avoid attributing performance differences to reimplementation. Evaluation
requires `conda activate cagean_tf` and a TF SavedModel checkpoint (Zenodo archive).
If the TF models were retrained on z-normalized epigenomic input (produced by
`data_prep/04_normalize.py`), pass `--norm_params data/norm_params_{mark}.json` to
`eval_tf_baseline.py`; if retrained on the same 0–1 range-normalized arrays as CAGEAN,
no additional normalization is needed.

**Zero-epi ablation (Supp Table S5).** `eval_zeroepi.py` zeroes the epigenomic input
to a trained CAGEAN at inference time. This is an out-of-distribution input (the model
never saw all-zero epi during training) and is reported for illustration only; it is
not a valid seq-only baseline. The proper seq-only baseline is the matched
`seqonly_transformer.pt` checkpoint trained without epigenomic input from the start.

---

## Citation

[pending]
