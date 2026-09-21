# Supplementary Figure S3: Layerwise representation probing of SeqOnlyTransformer
#
# Panel A: Token-level linear probe R² for three sequence features across the
#   CNN stem and four Transformer layers. Shows that nucleotide composition is
#   encoded at the input (CNN), while PQS distance is not decodable until
#   Layer 1 (R² ≈ 0.91).
# Panel B: Pooled class AUPRC (logistic probe on mean-pooled token embeddings)
#   across layers. Shows that G4+ vs G4− becomes linearly separable at Layer 3.
#
# Data source: analysis/run_probing.py → figures/data/figS3_probing_r2.csv
#                                         figures/data/figS3_probing_auprc.csv
# Run from project root: Rscript figures/figS3.R

source("figures/utils.R")

LAYER_LEVELS <- c("CNN", "Layer 1", "Layer 2", "Layer 3", "Layer 4")

PROBE_COLORS <- c(
  "dist(PQS)"   = unname(OI["vermil"]),
  "G fraction"  = unname(OI["blue"]),
  "GC fraction" = unname(OI["green"])
)

# ── Panel A: token-level R² ──────────────────────────────────────────────────
r2_dat <- read_csv("figures/data/figS3_probing_r2.csv", show_col_types = FALSE) |>
  filter(feature %in% names(PROBE_COLORS)) |>
  mutate(layer = factor(layer, levels = LAYER_LEVELS))

pA <- ggplot(r2_dat, aes(x = layer, y = score, color = feature,
                          group = feature, shape = feature)) +
  geom_hline(yintercept = 0, linetype = "dotted", linewidth = 0.3,
             color = "grey60") +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2.0) +
  scale_color_manual(values = PROBE_COLORS, name = NULL) +
  scale_shape_manual(
    values = c("dist(PQS)" = 16L, "G fraction" = 17L, "GC fraction" = 15L),
    name = NULL
  ) +
  scale_x_discrete(name = NULL) +
  scale_y_continuous(
    name   = expression(R^2~"(token-level linear probe)"),
    limits = c(-0.05, 1.0),
    breaks = seq(0, 1, 0.2)
  ) +
  labs(tag = "A") +
  theme_nar() +
  theme(legend.position = "bottom", legend.key.size = unit(0.35, "cm"))

# ── Panel B: pooled class AUPRC ──────────────────────────────────────────────
auprc_dat <- read_csv("figures/data/figS3_probing_auprc.csv",
                      show_col_types = FALSE) |>
  mutate(layer = factor(layer, levels = LAYER_LEVELS))

# positive rate as the random-classifier baseline
pos_rate <- auprc_dat |> filter(!is.na(pos_rate)) |> pull(pos_rate) |> first()

pB <- ggplot(auprc_dat, aes(x = layer, y = auprc, group = 1)) +
  geom_hline(yintercept = pos_rate, linetype = "dotted", linewidth = 0.3,
             color = "grey60") +
  annotate("text", x = 0.6, y = pos_rate + 0.012,
           label = "random", size = 2.2, color = "grey50", hjust = 0) +
  geom_line(linewidth = 0.6, color = OI["blue"]) +
  geom_point(size = 2.0, color = OI["blue"]) +
  scale_x_discrete(name = NULL) +
  scale_y_continuous(
    name   = "AUPRC (pooled logistic probe)",
    limits = c(NA, NA)
  ) +
  labs(tag = "B") +
  theme_nar()

p <- pA + pB + plot_layout(widths = c(1.1, 1))

save_fig(p, "figS3", width_cm = 14, height_cm = 7)
