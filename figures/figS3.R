# Supplementary Figure S3: Concatenation fusion ablation vs CAGEAN
# Line plot of Δ AUPRC (Concat+Res+InstanceNorm − CAGEAN) across evaluation
# datasets and epigenomic marks. Positive Δ: ablation outperforms CAGEAN.
# Negative Δ: CAGEAN outperforms ablation. H1975 H3K4me3 data not available.
#
# Run from project root: Rscript figures/figS3.R

source("figures/utils.R")

# cagean and concat_res are rounded to 3 d.p. from raw model outputs.
# delta is independently rounded from the raw difference (Table S12), so
# concat_res - cagean may differ from delta by ±0.001 in some rows.
# Only delta is used in the plot.
dat <- tribble(
  ~dataset,                    ~mark,     ~cagean, ~concat_res, ~delta,
  "A549\nsame-cell",           "H3K4me3",  0.886,   0.838,      -0.048,
  "A549\nsame-cell",           "H3K27ac",  0.888,   0.804,      -0.085,
  "A549\nsame-cell",           "ATAC",     0.849,   0.788,      -0.062,
  "HEK293T\ncross-cell",       "H3K4me3",  0.726,   0.698,      -0.028,
  "HEK293T\ncross-cell",       "H3K27ac",  0.665,   0.630,      -0.035,
  "HEK293T\ncross-cell",       "ATAC",     0.633,   0.621,      -0.012,
  "H1975\ncross-cell",         "H3K4me3",  NA,      NA,          NA,
  "H1975\ncross-cell",         "H3K27ac",  0.616,   0.639,      +0.023,
  "H1975\ncross-cell",         "ATAC",     0.624,   0.637,      +0.013,
  "K562 BG4\ncross-technique", "H3K4me3",  0.310,   0.336,      +0.025,
  "K562 BG4\ncross-technique", "H3K27ac",  0.318,   0.351,      +0.034,
  "K562 BG4\ncross-technique", "ATAC",     0.294,   0.348,      +0.053,
) |>
  mutate(
    dataset = factor(dataset, levels = unique(dataset)),
    mark    = factor(mark, levels = MARK_LEVELS)
  )

p <- ggplot(dat, aes(x = dataset, y = delta,
                     color = mark, shape = mark, group = mark)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4,
             color = "grey55") +
  geom_line(linewidth = 0.5, na.rm = TRUE,
            position = position_dodge(width = 0.25)) +
  geom_point(size = 2.2, na.rm = TRUE,
             position = position_dodge(width = 0.25)) +
  annotate("text", x = 4.5, y = 0.006,
           label = "Concatenation fusion better (above)",
           hjust = 1, vjust = 0, size = 2.5, color = "grey45") +
  annotate("text", x = 4.5, y = -0.006,
           label = "Cross-attention (CAGEAN) better (below)",
           hjust = 1, vjust = 1, size = 2.5, color = "grey45") +
  scale_color_manual(values = MARK_COLORS, name = NULL) +
  scale_shape_manual(
    values = c(H3K4me3 = 16L, H3K27ac = 17L, ATAC = 15L),
    name = NULL
  ) +
  scale_x_discrete(name = NULL) +
  scale_y_continuous(
    name   = "AUPRC (Concat+Res+InstanceNorm minus CAGEAN)",
    limits = c(-0.095, 0.065),
    breaks = seq(-0.09, 0.06, 0.03)
  ) +
  theme_nar() +
  theme(
    legend.position = "bottom",
    legend.key.size = unit(0.35, "cm")
  )

save_fig(p, "figS3", width_cm = 11, height_cm = 8)
