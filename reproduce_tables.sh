#!/usr/bin/env bash
# Reproduce all tables in the CAGEAN manuscript from Zenodo weights and arrays.
#
# Prerequisites:
#   1. conda activate cagean   (environment.yml)
#   2. Download the Zenodo archive and set ZENODO_DIR below.
#   3. For Tables 1 and 2 epiG4NN columns: activate cagean_tf and run the commands
#      printed at the end of this script.
#
# Runtime: ~1 hour on a single GPU.
set -euo pipefail

ZENODO_DIR=${1:-./zenodo}   # Pass as first argument or set here
CKPT=${ZENODO_DIR}/checkpoints
DATA=${ZENODO_DIR}/data
PQS=${ZENODO_DIR}/pqs/PQS_padded.bed
TSS=${ZENODO_DIR}/data/tss_2kb_hg19.bed

echo "================================================================"
echo " CAGEAN reproduction — reading from: ${ZENODO_DIR}"
echo "================================================================"

# ── Part 1: Table 1 and 2 — CAGEAN and Seq-only Transformer columns ──────────
echo ""
echo "=== Part 1: Table 1 / Table 2 (CAGEAN + Seq-only Transformer) ==="

# Seq-only Transformer baseline (Δarch decomposition rows)
echo "-- Seq-only Transformer A549 same-cell --"
python eval/eval_samecell.py \
    --model    seqonly \
    --ckpt     ${CKPT}/seqonly_transformer.pt \
    --data_dir ${DATA}/a549/test

echo "-- Seq-only Transformer cross-cell HEK293T + K562 BG4 --"
python eval/eval_crosscell.py \
    --model         seqonly \
    --ckpt          ${CKPT}/seqonly_transformer.pt \
    --crosscell_dir ${DATA}/crosscell \
    --pqs_bed       ${PQS}

# CAGEAN (one run per epigenomic mark)
for MARK in h3k4me3 h3k27ac atac; do
    echo "-- CAGEAN same-cell A549 ($MARK) --"
    python eval/eval_samecell.py \
        --ckpt     ${CKPT}/cagean_${MARK}.pt \
        --data_dir ${DATA}/a549/test \
        --mark     ${MARK}

    echo "-- CAGEAN cross-cell HEK293T + K562 BG4 ($MARK) --"
    python eval/eval_crosscell.py \
        --ckpt          ${CKPT}/cagean_${MARK}.pt \
        --crosscell_dir ${DATA}/crosscell \
        --pqs_bed       ${PQS} \
        --mark          ${MARK}
done

# ── Part 2: Table 3 — K562 BG4 same-cell ─────────────────────────────────────
echo ""
echo "=== Part 2: Table 3 (K562 BG4 same-cell) ==="
python eval/eval_bg4_samecell.py \
    --ckpt_seqonly ${CKPT}/bg4_seqonly_transformer.pt \
    --ckpt_h3k4me3 ${CKPT}/bg4_cagean_h3k4me3.pt \
    --ckpt_h3k27ac ${CKPT}/bg4_cagean_h3k27ac.pt \
    --ckpt_atac    ${CKPT}/bg4_cagean_atac.pt \
    --data_dir     ${DATA}/k562_bg4_samecell

# ── Part 3: Table 4 — stratified HEK293T ─────────────────────────────────────
echo ""
echo "=== Part 3: Table 4 (stratified HEK293T) ==="
python eval/eval_stratified.py \
    --ckpt_h3k4me3  ${CKPT}/cagean_h3k4me3.pt \
    --ckpt_h3k27ac  ${CKPT}/cagean_h3k27ac.pt \
    --ckpt_atac     ${CKPT}/cagean_atac.pt \
    --crosscell_dir ${DATA}/crosscell/hek293t \
    --pqs_bed       ${PQS} \
    --tss_bed       ${TSS}

# ── Part 4 (Tables 5 & 6): H9 ESC single-cell vs multi-cell ─────────────────
echo ""
echo "=== Part 4: Tables 5 and 6 (H9 ESC) ==="
for MARK in h3k4me3 h3k27ac atac; do
    python eval/eval_h9esc.py \
        --ckpt_single ${CKPT}/cagean_${MARK}.pt \
        --ckpt_multi  ${CKPT}/multicell_cagean_${MARK}.pt \
        --data_dir    ${DATA}/h9esc/test \
        --mark        ${MARK}
done

# ── Supp Table S5: zero-epi ablation ─────────────────────────────────────────
echo ""
echo "=== Supp Table S5 (zero-epi ablation) ==="
for MARK in h3k4me3 h3k27ac atac; do
    python eval/eval_zeroepi.py \
        --ckpt          ${CKPT}/cagean_${MARK}.pt \
        --data_dir      ${DATA}/a549/test \
        --crosscell_dir ${DATA}/crosscell/hek293t \
        --pqs_bed       ${PQS} \
        --mark          ${MARK}
done

# ── Supp Table S10: constitutive vs. strict CTS stratification ───────────────
echo ""
echo "=== Supp Table S10 (constitutive vs. strict CTS) ==="
python eval/eval_strict_cts.py \
    --ckpt_h3k4me3  ${CKPT}/cagean_h3k4me3.pt \
    --ckpt_h3k27ac  ${CKPT}/cagean_h3k27ac.pt \
    --ckpt_atac     ${CKPT}/cagean_atac.pt \
    --ckpt_seqonly  ${CKPT}/seqonly_transformer.pt \
    --a549_dir      ${DATA}/a549/test \
    --crosscell_dir ${DATA}/crosscell/hek293t \
    --pqs_bed       ${PQS}

# ── Supp Table S12: concatenation-fusion ablation vs CAGEAN ─────────────────
echo ""
echo "=== Supp Table S12 (concatenation-fusion ablation) ==="
python eval/eval_ablation.py \
    --ckpt_cagean_h3k4me3   ${CKPT}/cagean_h3k4me3.pt \
    --ckpt_cagean_h3k27ac   ${CKPT}/cagean_h3k27ac.pt \
    --ckpt_cagean_atac      ${CKPT}/cagean_atac.pt \
    --ckpt_ablation_h3k4me3 ${CKPT}/concat_fusion_h3k4me3.pt \
    --ckpt_ablation_h3k27ac ${CKPT}/concat_fusion_h3k27ac.pt \
    --ckpt_ablation_atac    ${CKPT}/concat_fusion_atac.pt \
    --a549_dir              ${DATA}/a549/test \
    --crosscell_dir         ${DATA}/crosscell \
    --pqs_bed               ${PQS}
# Add --h1975_dir ${DATA}/crosscell/h1975 --q_h1975 0.07343 if H1975 arrays are available

# ── Tables 1 & 2 epiG4NN columns (requires cagean_tf environment) ────────────
echo ""
echo "=== epiG4NN TF baseline (Tables 1 & 2) ==="
echo "    NOTE: activate cagean_tf before running:"
echo "    conda activate cagean_tf"
echo "    Then re-run this section manually:"
echo ""
for MARK in h3k4me3 h3k27ac atac; do
    echo "    python eval/eval_tf_baseline.py \\"
    echo "        --mark          ${MARK} \\"
    echo "        --ckpt_dir      ${CKPT}/tf_epig4nn_${MARK} \\"
    echo "        --a549_test_dir ${DATA}/a549/test \\"
    echo "        --crosscell_dir ${DATA}/crosscell/hek293t \\"
    echo "        --pqs_bed       ${PQS}"
    echo "    # Add --ckpt_seqonly path if you have a seq-only epiG4NN checkpoint"
    echo "    # (optional; needed for the Δ(arch) decomposition row in Table 2)"
    echo ""
done

echo "================================================================"
echo " All PyTorch eval complete."
echo " Run TF section above in cagean_tf environment to complete Tables 1-2."
echo "================================================================"
