# Supplementary Figure S1: Training protocol comparison
# Panel: AUPRC across chromosome-sequential and epoch-based training protocols
#
# Run from project root: Rscript figures/figS1.R

source("figures/utils.R")

s2 <- tribble(
  ~protocol,                              ~mark,     ~a549,  ~hek,
  "Chrom-sequential\nstep-decay",         "H3K4me3",  0.866,  0.724,
  "Chrom-sequential\nstep-decay",         "H3K27ac",  0.882,  0.665,
  "Chrom-sequential\nstep-decay",         "ATAC",     0.816,  0.637,
  "Chrom-sequential\nAdamW+cosine",       "H3K4me3",  0.876,  0.728,
  "Chrom-sequential\nAdamW+cosine",       "H3K27ac",  0.882,  0.661,
  "Chrom-sequential\nAdamW+cosine",       "ATAC",     0.822,  0.644,
  "Epoch-based\nAdamW+cosine\n(CAGEAN)",  "H3K4me3",  0.895,  0.726,
  "Epoch-based\nAdamW+cosine\n(CAGEAN)",  "H3K27ac",  0.895,  0.665,
  "Epoch-based\nAdamW+cosine\n(CAGEAN)",  "ATAC",     0.852,  0.633,
) |>
  pivot_longer(c(a549, hek), names_to = "eval", values_to = "auprc") |>
  mutate(
    eval     = recode(eval, a549 = "A549 same-cell", hek = "HEK293T cross-cell"),
    mark     = factor(mark, levels = MARK_LEVELS),
    protocol = factor(protocol, levels = unique(protocol))
  )

proto_colors <- c(
  "Chrom-sequential\nstep-decay"        = unname(OI["gray"]),
  "Chrom-sequential\nAdamW+cosine"      = unname(OI["orange"]),
  "Epoch-based\nAdamW+cosine\n(CAGEAN)" = unname(OI["blue"])
)

p <- ggplot(s2, aes(x = mark, y = auprc, fill = protocol)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7,
           color = "grey20", linewidth = 0.25) +
  facet_wrap(~eval, nrow = 1) +
  scale_fill_manual(values = proto_colors, name = NULL) +
  scale_y_continuous(name = "AUPRC", limits = c(0, 1.02), breaks = seq(0, 1, 0.2),
                     expand = expansion(mult = c(0, 0.02))) +
  scale_x_discrete(name = NULL) +
  theme_nar() +
  theme(
    legend.position  = "bottom",
    legend.key.size  = unit(0.25, "cm")
  )

save_fig(p, "figS1", width_cm = 17.4, height_cm = 6)
