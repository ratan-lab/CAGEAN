# Figure 2: Decomposition of performance gains and comparison of assay types
# Panels: (A) Absolute AUPRC decomposition; (B) K562 BG4 same-cell AUPRC;
#         (C) figure2c.svg; (D) ISM inside/outside PQS importance
#
# Run from project root: Rscript figures/fig2.R

source("figures/utils.R")

# ── Panel A: Δ(arch) + Δ(epi) decomposition ─────────────────────────────────

t2 <- tribble(
  ~dataset,                   ~mark,     ~seq_cnn, ~seq_transformer, ~cagean,
  "A549\nsame-cell",          "H3K4me3",  0.603,    0.726,            0.895,
  "A549\nsame-cell",          "H3K27ac",  0.603,    0.726,            0.895,
  "A549\nsame-cell",          "ATAC",     0.603,    0.726,            0.852,
  "HEK293T\ncross-cell",      "H3K4me3",  0.548,    0.627,            0.726,
  "HEK293T\ncross-cell",      "H3K27ac",  0.548,    0.627,            0.665,
  "HEK293T\ncross-cell",      "ATAC",     0.548,    0.627,            0.633,
  "H1975\ncross-cell",        "H3K27ac",  0.520,    0.591,            0.616,
  "H1975\ncross-cell",        "ATAC",     0.520,    0.591,            0.624,
  "K562 BG4\n(A549-trained)", "H3K4me3",  0.286,    0.292,            0.310,
  "K562 BG4\n(A549-trained)", "H3K27ac",  0.286,    0.292,            0.318,
  "K562 BG4\n(A549-trained)", "ATAC",     0.286,    0.292,            0.294,
  "U2OS BG4\n(A549-trained)", "H3K4me3",  0.113,    0.146,            0.119,
  "U2OS BG4\n(A549-trained)", "H3K27ac",  0.113,    0.146,            0.128,
  "U2OS BG4\n(A549-trained)", "ATAC",     0.113,    0.146,            0.125,
) |>
  mutate(
    dataset = factor(dataset, levels = unique(dataset)),
    mark    = factor(mark, levels = MARK_LEVELS)
  )

t2_long <- t2 |>
  select(dataset, mark, seq_cnn, seq_transformer, cagean) |>
  pivot_longer(c(seq_cnn, seq_transformer, cagean),
               names_to = "model", values_to = "auprc") |>
  mutate(
    model = factor(recode(model,
                          seq_cnn         = "Seq-only CNN",
                          seq_transformer = "Seq-only Transformer",
                          cagean          = "CAGEAN"),
                   levels = c("Seq-only CNN", "Seq-only Transformer", "CAGEAN"))
  )

comp_colors <- c(
  "Seq-only CNN"         = unname(OI["gray"]),
  "Seq-only Transformer" = unname(OI["sky"]),
  "CAGEAN"               = unname(OI["blue"])
)

p_a <- ggplot(t2_long, aes(x = mark, y = auprc, fill = model)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65,
           color = "grey20", linewidth = 0.25) +
  facet_grid(~dataset, scales = "free_x", space = "free_x") +
  scale_fill_manual(values = comp_colors, name = NULL) +
  scale_y_continuous(name = "AUPRC", limits = c(0, 1.02),
                     breaks = seq(0, 1, 0.2),
                     expand = expansion(mult = c(0, 0.02))) +
  scale_x_discrete(name = NULL) +
  theme_nar() +
  theme(
    legend.position = "bottom",
    axis.text.x     = element_text(size = 6.5, angle = 40, hjust = 1, vjust = 1)

  ) +
  labs(tag = "A")

# ── Panel B: K562 BG4 same-cell AUPRC ───────────────────────────────────────

t3 <- tribble(
  ~model,                ~mark,         ~auprc, ~d_epi,   ~ci_lo,  ~ci_hi,
  "Seq-only Transformer", NA_character_, 0.337,  NA_real_, NA_real_, NA_real_,
  "CAGEAN",              "H3K4me3",     0.427,  +0.090,  +0.086,  +0.093,
  "CAGEAN",              "H3K27ac",     0.434,  +0.097,  +0.093,  +0.101,
  "CAGEAN",              "ATAC",        0.492,  +0.155,  +0.151,  +0.158,
) |>
  mutate(
    x_label = if_else(is.na(mark), "Seq-only\nTransformer", paste0("CAGEAN\n", mark)),
    x_label = factor(x_label, levels = c("Seq-only\nTransformer",
                                          "CAGEAN\nH3K4me3",
                                          "CAGEAN\nH3K27ac",
                                          "CAGEAN\nATAC")),
    fill_col = if_else(is.na(mark), "Seq-only Transformer", mark),
    fill_col = factor(fill_col, levels = c("Seq-only Transformer", MARK_LEVELS))
  )

bar_colors_2b <- c(
  "Seq-only Transformer" = unname(OI["sky"]),
  MARK_COLORS
)

p_b <- ggplot(t3, aes(x = x_label, y = auprc, fill = fill_col)) +
  geom_col(width = 0.65, color = "grey20", linewidth = 0.25) +
  geom_errorbar(
    data = filter(t3, !is.na(d_epi)),
    aes(ymin = auprc + ci_lo - d_epi, ymax = auprc + ci_hi - d_epi),
    width = 0.22, linewidth = 0.5, color = "grey20"
  ) +
  geom_text(
    data = filter(t3, !is.na(d_epi)),
    aes(y = auprc + ci_hi - d_epi + 0.018,
        label = sprintf("%+.3f", d_epi)),
    size = 2.3, color = "grey20"
  ) +
  scale_fill_manual(values = bar_colors_2b, name = NULL) +
  scale_y_continuous(name = "AUPRC (K562 BG4 same-cell)",
                     limits = c(0, 0.63), breaks = seq(0, 0.6, 0.1),
                     expand = expansion(mult = c(0, 0.02))) +
  scale_x_discrete(name = NULL) +
  theme_nar() +
  theme(legend.position = "none",
        axis.text.x     = element_text(size = 6.5, angle = 40, hjust = 1, vjust = 1)) +
  labs(tag = "B")

# ── Panel C: assay comparison diagram ───────────────────────────────────────
p_c <- svg_to_panel("figure2c.svg", tag = "C")

# ── Panel D: ISM inside/outside PQS importance ───────────────────────────────
ism_dat <- read_csv("figures/data/ism_in_out.csv", show_col_types = FALSE) |>
  mutate(
    class  = factor(if_else(class == "pos", "G4+", "G4-"),
                    levels = c("G4+", "G4-")),
    region = factor(if_else(region == "inside", "Inside PQS", "Flanking"),
                    levels = c("Inside PQS", "Flanking"))
  )

ISM_COLORS <- c(
  "Inside PQS" = unname(OI["vermil"]),
  "Flanking"   = unname(OI["blue"])
)

ism_annot <- ism_dat |>
  group_by(class) |>
  summarise(y = max(imp) * 1.10, .groups = "drop") |>
  mutate(label = c("1.54x", "1.97x"))

p_d <- ggplot(ism_dat, aes(x = region, y = imp, fill = region, color = region)) +
  geom_violin(alpha = 0.25, linewidth = 0.3, trim = TRUE, scale = "width") +
  geom_boxplot(width = 0.2, outlier.shape = NA, linewidth = 0.35,
               fill = "white", color = "grey30") +
  geom_text(data = ism_annot,
            aes(x = 1.5, y = y, label = label),
            inherit.aes = FALSE,
            size = 2.5, fontface = "bold", color = "grey20") +
  facet_wrap(~class, ncol = 2) +
  scale_fill_manual(values = ISM_COLORS, guide = "none") +
  scale_color_manual(values = ISM_COLORS, guide = "none") +
  scale_x_discrete(name = NULL) +
  scale_y_continuous(
    name   = "Mean |delta-logit| per sequence",
    expand = expansion(mult = c(0.02, 0.14))
  ) +
  labs(tag = "D") +
  theme_nar() +
  theme(
    strip.text  = element_text(face = "bold", size = 7.5),
    axis.text.x = element_text(size = 7)
  )

# ── Assemble and save ────────────────────────────────────────────────────────

top_row <- p_a + p_b + plot_layout(widths = c(2.2, 1))
bot_row <- p_c + p_d + plot_layout(widths = c(1.5, 1))

fig2 <- top_row / bot_row +
  plot_layout(heights = c(1, 1)) &
  theme(plot.tag.position = c(0, 1), plot.tag = element_text(face = "bold", size = 9))

save_fig(fig2, "fig2", width_cm = 17.4, height_cm = 15)
