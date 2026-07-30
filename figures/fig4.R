# Figure 4: Stratified analysis and cell-type distance gradient
# Panels: (A) HEK293T test set stratum composition; (B) stratified AUPRC (HEK293T);
#         (C) promoter/enhancer distance gradient; (D) multi-cell training comparison
#
# Run from project root: Rscript figures/fig4.R

source("figures/utils.R")

# ── Panel A: HEK293T stratum composition ────────────────────────────────────

strata <- tribble(
  ~stratum,   ~n,       ~pos_pct,
  "Promoter",  58938,    40.8,
  "Enhancer",  54300,     7.9,
  "Inactive", 234764,     2.7
) |>
  mutate(
    stratum = factor(stratum, levels = c("Promoter", "Enhancer", "Inactive")),
    pct_all = n / sum(n) * 100
  )

p_a <- ggplot(strata, aes(x = stratum, y = pos_pct, fill = stratum)) +
  geom_col(width = 0.6, color = "grey20", linewidth = 0.25) +
  geom_text(aes(label = paste0(pos_pct, "%")),
            vjust = -0.3, size = 2.8, color = "grey20") +
  scale_fill_manual(values = STRATUM_COLORS, name = NULL) +
  scale_y_continuous(name = "G4-positive rate (%)",
                     limits = c(0, 50), expand = expansion(mult = c(0, 0.05))) +
  scale_x_discrete(
    name   = NULL,
    labels = c(
      "Promoter" = "Promoter\n(n=58,938)",
      "Enhancer" = "Enhancer\n(n=54,300)",
      "Inactive" = "Inactive\n(n=234,764)"
    )
  ) +
  theme_nar() +
  theme(
    legend.position   = "none",
    plot.tag.position = c(0, 1),
    plot.tag          = element_text(face = "bold", size = 9, hjust = 0)
  ) +
  labs(tag = "A")

# ── Panel B: stratified AUPRC (HEK293T cross-cell) ──────────────────────────

t4 <- tribble(
  ~stratum,     ~mark,     ~cagean, ~seq_transformer,
  "All sites",  "H3K4me3",  0.726,   0.627,
  "All sites",  "H3K27ac",  0.665,   0.627,
  "All sites",  "ATAC",     0.633,   0.627,
  "Promoter",   "H3K4me3",  0.869,   0.766,
  "Promoter",   "H3K27ac",  0.823,   0.766,
  "Promoter",   "ATAC",     0.783,   0.766,
  "Enhancer",   "H3K4me3",  0.437,   0.371,
  "Enhancer",   "H3K27ac",  0.376,   0.371,
  "Enhancer",   "ATAC",     0.363,   0.371,
) |>
  mutate(
    stratum = factor(stratum, levels = c("All sites", "Promoter", "Enhancer")),
    mark    = factor(mark, levels = MARK_LEVELS)
  )

p_b <- ggplot(t4, aes(x = mark, y = cagean, fill = mark)) +
  geom_col(width = 0.65, color = "grey20", linewidth = 0.25) +
  geom_point(aes(y = seq_transformer), shape = 23, fill = "white",
             color = "grey30", size = 1.5, stroke = 0.4) +
  facet_wrap(~stratum, nrow = 1, scales = "free_y") +
  scale_fill_manual(values = MARK_COLORS, name = NULL) +
  scale_y_continuous(name = "AUPRC (HEK293T cross-cell)",
                     expand = expansion(mult = c(0, 0.06))) +
  scale_x_discrete(name = NULL) +
  theme_nar() +
  theme(
    legend.position = "none",
    axis.text.x     = element_text(angle = 30, hjust = 1, size = 8)
  ) +
  labs(
    tag     = "B"
  )

# ── Panel C: cell-type distance gradient ─────────────────────────────────────

t5 <- tribble(
  ~cell_type, ~stratum,   ~mark,     ~auprc,
  "A549",     "Promoter", "H3K4me3",  0.933,
  "A549",     "Promoter", "H3K27ac",  0.938,
  "A549",     "Promoter", "ATAC",     0.899,
  "A549",     "Enhancer", "H3K4me3",  0.749,
  "A549",     "Enhancer", "H3K27ac",  0.731,
  "A549",     "Enhancer", "ATAC",     0.703,
  "HEK293T",  "Promoter", "H3K4me3",  0.869,
  "HEK293T",  "Promoter", "H3K27ac",  0.823,
  "HEK293T",  "Promoter", "ATAC",     0.783,
  "HEK293T",  "Enhancer", "H3K4me3",  0.437,
  "HEK293T",  "Enhancer", "H3K27ac",  0.376,
  "HEK293T",  "Enhancer", "ATAC",     0.363,
  "H9 ESC",   "Promoter", "H3K4me3",  0.114,
  "H9 ESC",   "Promoter", "H3K27ac",  0.183,
  "H9 ESC",   "Promoter", "ATAC",     0.125,
  "H9 ESC",   "Enhancer", "H3K4me3",  0.120,
  "H9 ESC",   "Enhancer", "H3K27ac",  0.072,
  "H9 ESC",   "Enhancer", "ATAC",     0.062,
) |>
  mutate(
    cell_type = factor(cell_type, levels = CELL_LEVELS),
    stratum   = factor(stratum, levels = c("Promoter", "Enhancer")),
    mark      = factor(mark, levels = MARK_LEVELS)
  )

p_c <- ggplot(t5, aes(x = cell_type, y = auprc,
                       color = mark, group = interaction(mark, stratum),
                       linetype = stratum, shape = stratum)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2) +
  scale_color_manual(values = MARK_COLORS, name = "Mark") +
  scale_linetype_manual(values = c(Promoter = "solid", Enhancer = "dashed"),
                        name = "Stratum") +
  scale_shape_manual(values = c(Promoter = 16, Enhancer = 17),
                     name = "Stratum") +
  scale_x_discrete(name = NULL) +
  scale_y_continuous(name = "AUPRC", breaks = seq(0, 1, 0.2)) +
  theme_nar() +
  theme(legend.position = "bottom") +
  labs(tag = "C")

# ── Panel D: multi-cell training comparison (H9 ESC) ────────────────────────

t6 <- tribble(
  ~mark,     ~a549_only, ~multicell,
  "H3K4me3",  0.114,      0.109,
  "H3K27ac",  0.140,      0.120,
  "ATAC",     0.079,      0.079,
) |>
  pivot_longer(c(a549_only, multicell), names_to = "training", values_to = "auprc") |>
  mutate(
    training = recode(training,
                      a549_only = "A549 only",
                      multicell = "Multi-cell\n(A549+HeLa.S3+HepG2)"),
    training = factor(training, levels = c("A549 only",
                                           "Multi-cell\n(A549+HeLa.S3+HepG2)")),
    mark = factor(mark, levels = MARK_LEVELS)
  )

p_d <- ggplot(t6, aes(x = mark, y = auprc, fill = training)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6,
           color = "grey20", linewidth = 0.25) +
  scale_fill_manual(
    values = c("A549 only"                       = unname(OI["blue"]),
               "Multi-cell\n(A549+HeLa.S3+HepG2)" = unname(OI["orange"])),
    name = NULL
  ) +
  scale_y_continuous(
    name   = "AUPRC (H9 ESC all sites)",
    limits = c(0, 0.20),
    breaks = seq(0, 0.2, 0.05),
    expand = expansion(mult = c(0, 0.05))
  ) +
  scale_x_discrete(name = NULL) +
  theme_nar() +
  theme(legend.position = "bottom") +
  labs(tag = "D")

# ── Assemble and save ────────────────────────────────────────────────────────

top_row    <- p_a + p_b + plot_layout(widths = c(0.7, 1.3))
bottom_row <- p_c + p_d + plot_layout(widths = c(1.4, 1))

fig4 <- top_row / bottom_row

save_fig(fig4, "fig4", width_cm = 17.4, height_cm = 12)
