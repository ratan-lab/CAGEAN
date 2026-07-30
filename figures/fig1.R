# Figure 1: CAGEAN architecture, domain-shift diagnosis, and performance overview
# Panels: (A) architecture diagram; (B) signal distributions; (C) diagnostic ΔAUPRC;
#         (D) epiG4NN vs CAGEAN performance comparison
#
# Run from project root: Rscript figures/fig1.R
# Requires: figure1a.svg, figures/data/fig1b.csv

source("figures/utils.R")

# ── Panel a: architecture diagram ───────────────────────────────────────────
p_a <- svg_to_panel("figure1a.svg", tag = "A")

# ── Panel b: signal distributions at G4-positive loci ──────────────────────

df_sig <- read_csv("figures/data/fig1b.csv", show_col_types = FALSE) |>
  mutate(
    mark      = factor(mark, levels = MARK_LEVELS),
    cell_type = factor(cell_type, levels = c("A549", "HEK293T"))
  )

# Fold-change annotation for each mark (A549 mean / HEK293T mean)
fc_labels <- df_sig |>
  group_by(mark, cell_type) |>
  summarise(mn = mean(mean_signal), .groups = "drop") |>
  pivot_wider(names_from = cell_type, values_from = mn) |>
  mutate(label = paste0("~", round(A549 / HEK293T), "×"))

p_b <- ggplot(df_sig, aes(x = cell_type, y = mean_signal + 1e-6, fill = cell_type)) +
  geom_violin(scale = "width", trim = TRUE, linewidth = 0.3, alpha = 0.85) +
  geom_boxplot(width = 0.12, outlier.shape = NA, linewidth = 0.3,
               fill = "white", color = "grey30") +
  geom_text(
    data = fc_labels,
    aes(x = 1.5, y = Inf, label = label),
    inherit.aes = FALSE,
    vjust = 1.4, size = 2.8, color = "grey30"
  ) +
  facet_wrap(~mark, scales = "free_y", nrow = 1) +
  scale_y_log10(
    labels = scales::label_log(),
    name   = "Mean signal per site (log₁₀)"
  ) +
  scale_fill_manual(values = CELL_COLORS, name = NULL) +
  scale_x_discrete(name = NULL) +
  theme_nar() +
  theme(
    legend.position  = "bottom",
    axis.text.x      = element_text(size = 6.5, angle = 40, hjust = 1, vjust = 1)
  ) +
  labs(tag = "B")

# ── Panel c: diagnostic delta AUPRC ─────────────────────────────────────────

diag <- tribble(
  ~architecture,         ~eval_cell, ~mark,     ~delta_auprc,
  "Batch-norm + concat", "HEK293T",  "H3K4me3",  +0.001,
  "Batch-norm + concat", "HEK293T",  "H3K27ac",  +0.013,
  "Batch-norm + concat", "HEK293T",  "ATAC",     +0.111,
  "CAGEAN",              "HEK293T",  "H3K4me3",  +0.001,
  "CAGEAN",              "HEK293T",  "H3K27ac",  -0.002,
  "CAGEAN",              "HEK293T",  "ATAC",     +0.032,
  "Batch-norm + concat", "H1975",    "H3K27ac",  -0.006,
  "Batch-norm + concat", "H1975",    "ATAC",     -0.007,
  "CAGEAN",              "H1975",    "H3K27ac",  -0.009,
  "CAGEAN",              "H1975",    "ATAC",     -0.037,
) |>
  mutate(
    architecture = factor(architecture, levels = c("Batch-norm + concat", "CAGEAN")),
    eval_cell    = factor(eval_cell, levels = c("HEK293T", "H1975")),
    mark         = factor(mark, levels = MARK_LEVELS)
  )

p_c <- ggplot(diag, aes(x = mark, y = delta_auprc, fill = architecture)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6, linewidth = 0.25,
           color = "grey20") +
  geom_hline(yintercept = 0, linewidth = 0.35, color = "grey40") +
  facet_wrap(~eval_cell, nrow = 1, scales = "free_x") +
  scale_fill_manual(
    values = c("Batch-norm + concat" = unname(OI["orange"]),
               "CAGEAN"             = unname(OI["blue"])),
    name = NULL
  ) +
  scale_y_continuous(
    name   = "ΔAUPRC (cell-type-specific\nnormalization at inference)",
    limits = c(-0.05, 0.125),
    breaks = c(-0.04, 0, 0.04, 0.08, 0.12)
  ) +
  scale_x_discrete(name = NULL, drop = TRUE) +
  theme_nar() +
  theme(
    legend.position  = "bottom",
    axis.text.x      = element_text(size = 6.5, angle = 40, hjust = 1, vjust = 1)
  ) +
  labs(tag = "C")

# ── Panel D: epiG4NN vs CAGEAN across datasets ──────────────────────────────

t1 <- tribble(
  ~dataset,               ~mark,     ~epiG4NN,  ~CAGEAN,
  "A549\nsame-cell",      "H3K4me3",  0.860,     0.895,
  "A549\nsame-cell",      "H3K27ac",  0.846,     0.895,
  "A549\nsame-cell",      "ATAC",     0.775,     0.852,
  "HEK293T\ncross-cell",  "H3K4me3",  0.731,     0.726,
  "HEK293T\ncross-cell",  "H3K27ac",  0.617,     0.665,
  "HEK293T\ncross-cell",  "ATAC",     0.560,     0.633,
  "H1975\ncross-cell",    "H3K27ac",  0.615,     0.616,
  "H1975\ncross-cell",    "ATAC",     0.617,     0.624,
  "K562\ncross-technique","H3K4me3",  0.306,     0.310,
  "K562\ncross-technique","H3K27ac",  0.324,     0.318,
  "K562\ncross-technique","ATAC",     0.328,     0.294,
  "U2OS BG4\n(cross-cell+tech)","H3K4me3", NA_real_, 0.119,
  "U2OS BG4\n(cross-cell+tech)","H3K27ac", NA_real_, 0.128,
  "U2OS BG4\n(cross-cell+tech)","ATAC",    NA_real_, 0.125,
) |>
  pivot_longer(c(epiG4NN, CAGEAN), names_to = "model", values_to = "auprc") |>
  mutate(
    dataset = factor(dataset,
                     levels = c("A549\nsame-cell",
                                "HEK293T\ncross-cell",
                                "H1975\ncross-cell",
                                "K562\ncross-technique",
                                "U2OS BG4\n(cross-cell+tech)")),
    mark    = factor(mark, levels = MARK_LEVELS),
    model   = factor(model, levels = c("epiG4NN", "CAGEAN"))
  )

p_d <- ggplot(t1, aes(x = dataset, y = auprc, fill = model)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65,
           color = "grey20", linewidth = 0.25) +
  facet_wrap(~mark, nrow = 1, scales = "free_x") +
  scale_fill_manual(
    values = c("epiG4NN" = unname(OI["purple"]), "CAGEAN" = unname(OI["blue"])),
    name   = NULL
  ) +
  scale_y_continuous(name = "AUPRC", limits = c(0, 1.02), breaks = seq(0, 1, 0.2),
                     expand = expansion(mult = c(0, 0.02))) +
  scale_x_discrete(name = NULL) +
  theme_nar() +
  theme(
    legend.position  = "bottom",
    axis.text.x      = element_text(size = 6.5, angle = 40, hjust = 1, vjust = 1)
  ) +
  labs(tag = "D")

# ── Assemble and save ────────────────────────────────────────────────────────

fig1 <- p_a / (p_b + p_c + plot_layout(widths = c(1.4, 1))) / p_d +
  plot_layout(heights = c(2.5, 1, 1)) &
  theme(plot.tag.position = c(0, 1), plot.tag = element_text(face = "bold", size = 9))

save_fig(fig1, "fig1", width_cm = 17.4, height_cm = 20)
