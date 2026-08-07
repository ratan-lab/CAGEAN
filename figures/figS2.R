# Supplementary Figure S2: K562 cross-technique AUPRC ceiling across marks
# Scatter plot: A549 same-cell AUPRC (x) vs K562 BG4 cross-technique AUPRC (y)
# for 10 epigenomic marks (3 reference histone/ATAC + 7 additional marks),
# all evaluated under the same T+NZ Q90 protocol. Spread in same-cell AUPRC
# is ~18 percentage points; spread in cross-technique AUPRC is ~3 pp.
#
# Run from project root: Rscript figures/figS2.R

library(ggrepel)
source("figures/utils.R")

# Seq-only Transformer K562 cross-technique AUPRC (Table 2; corrected from bootstrap)
SEQ_TRANSFORMER_K562 <- 0.292

dat <- tribble(
  ~mark,       ~group,             ~same_cell, ~k562_cross,
  "H3K4me3",   "Reference",        0.8946,     0.3198,
  "H3K27ac",   "Reference",        0.8947,     0.3266,
  "ATAC",      "Reference",        0.8521,     0.3057,
  "SP1",       "Additional marks", 0.7246,     0.3068,
  "POLR2A",    "Additional marks", 0.7189,     0.3054,
  "BRD4",      "Additional marks", 0.7970,     0.2987,
  "H3K27me3",  "Additional marks", 0.7292,     0.3063,
  "DNase",     "Additional marks", 0.7263,     0.3066,
  "MYC",       "Additional marks", 0.7185,     0.3036,
  "EP300",     "Additional marks", 0.7411,     0.3049,
) |>
  mutate(group = factor(group, levels = c("Reference", "Additional marks")))

group_colors <- c(
  "Reference"        = unname(OI["blue"]),
  "Additional marks" = unname(OI["orange"])
)

group_shapes <- c(
  "Reference"        = 16L,
  "Additional marks" = 17L
)

p <- ggplot(dat, aes(x = same_cell, y = k562_cross,
                     color = group, shape = group, label = mark)) +
  # Seq-only Transformer reference line
  geom_hline(yintercept = SEQ_TRANSFORMER_K562,
             linetype = "dashed", linewidth = 0.4, color = unname(OI["sky"])) +
  annotate("text", x = 0.905, y = SEQ_TRANSFORMER_K562 - 0.004,
           label = "Seq-only Transformer", hjust = 1,
           size = 2.8, color = unname(OI["sky"])) +
  geom_point(size = 2.0) +
  geom_text_repel(
    size               = 2.8,
    box.padding        = 0.3,
    point.padding      = 0.2,
    min.segment.length = 0.2,
    segment.size       = 0.3,
    max.overlaps       = Inf,
    show.legend        = FALSE
  ) +
  scale_color_manual(values = group_colors, name = NULL) +
  scale_shape_manual(values = group_shapes, name = NULL) +
  scale_x_continuous(
    name   = "A549 same-cell AUPRC",
    limits = c(0.695, 0.910),
    breaks = seq(0.70, 0.90, 0.05)
  ) +
  scale_y_continuous(
    name   = "K562 BG4 cross-technique AUPRC (T+NZ)",
    limits = c(0.278, 0.340),
    breaks = seq(0.28, 0.34, 0.02),
    expand = expansion(mult = c(0.04, 0.04))
  ) +
  theme_nar() +
  theme(
    legend.position    = c(0.78, 0.50),
    legend.key.size    = unit(0.3, "cm"),
    panel.grid.major.x = element_line(color = "grey90", linewidth = 0.25)
  )

save_fig(p, "figS2", width_cm = 8.5, height_cm = 8)
