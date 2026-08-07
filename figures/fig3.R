# Figure 3: Cross-attention weight profiles by epigenomic mark and genomic group
# Panels: (A) marginal attention profiles (all 3 marks); (B) row entropy + GGG density overlay;
#         (C) attention heatmaps (H3K27ac)
#
# Run from project root: Rscript figures/fig3.R
# Requires: figures/data/fig4a.csv, fig4b.csv, fig4c.csv, fig4c_ggg.csv

source("figures/utils.R")

# ── Panel (A) groups shown per legend ───────────────────────────────────────
PANEL_A_GROUPS <- c(
  "A549 G4+ promoter", "A549 G4+ enhancer",
  "HEK293T G4+ promoter", "HEK293T G4+ enhancer"
)

# ── Panel A: marginal attention profiles ────────────────────────────────────

df_a <- read_csv("figures/data/fig4a.csv", show_col_types = FALSE) |>
  filter(group %in% PANEL_A_GROUPS) |>
  mutate(
    mark  = factor(mark, levels = c("H3K4me3", "H3K27ac", "ATAC")),
    group = factor(group, levels = names(ATTN_GROUP_COLORS))
  )

make_marg_panel <- function(data, mark_name, y_lab = TRUE) {
  ggplot(data |> filter(mark == mark_name),
         aes(x = position_bp, y = epi_marg, color = group)) +
    geom_line(linewidth = 0.55) +
    scale_color_manual(values = ATTN_GROUP_COLORS, name = NULL,
                       drop = TRUE) +
    scale_x_continuous(
      name   = "Position in window (bp)",
      limits = c(0, 1000), breaks = seq(0, 1000, 250)
    ) +
    scale_y_continuous(
      name   = if (y_lab) "Marginal attention" else NULL,
      limits = c(0, 0.11),
      breaks = c(0, 0.05, 0.10),
      labels = scales::label_number(accuracy = 0.01)
    ) +
    annotate("text", x = 5, y = Inf, label = mark_name,
             hjust = 0, vjust = 1.3, size = 2.8, fontface = "bold") +
    theme_nar() +
    theme(legend.key.height = unit(0.35, "cm"))
}

p_a_h3k4me3 <- make_marg_panel(df_a, "H3K4me3") +
  labs(tag = "A") +
  theme(plot.tag.position = c(0, 1),
        plot.tag          = element_text(face = "bold", size = 9, hjust = 0))
p_a_h3k27ac <- make_marg_panel(df_a, "H3K27ac")
p_a_atac    <- make_marg_panel(df_a, "ATAC")

p_a <- (p_a_h3k4me3 + p_a_h3k27ac + p_a_atac) +
  plot_layout(guides = "collect") &
  theme(
    legend.position   = "bottom",
    legend.key.height = unit(0.35, "cm"),
    plot.margin       = margin(2, 2, 2, 2)
  )
p_a <- p_a & guides(color = guide_legend(nrow = 2, byrow = TRUE))

# ── Panel B: attention heatmaps ─────────────────────────────────────────────

HEATMAP_GROUPS <- c("A549 G4+ promoter", "A549 G4+ enhancer",
                    "HEK293T G4+ promoter", "HEK293T G4+ enhancer")
HEATMAP_TITLES <- c(
  "A549 G4+ promoter"    = "A549\nG4+ promoter",
  "A549 G4+ enhancer"    = "A549\nG4+ enhancer",
  "HEK293T G4+ promoter" = "HEK293T\nG4+ promoter",
  "HEK293T G4+ enhancer" = "HEK293T\nG4+ enhancer"
)

df_b <- read_csv("figures/data/fig4b.csv", show_col_types = FALSE) |>
  mutate(
    group     = factor(group, levels = HEATMAP_GROUPS),
    log_attn  = log10(attn_weight + 1e-6),
    seq_pos   = (seq_bin + 0.5) * 10,
    epi_pos   = (epi_bin + 0.5) * 10
  )

attn_lim <- quantile(df_b$log_attn, c(0.01, 0.99))

make_heatmap <- function(data, grp) {
  ggplot(data |> filter(group == grp),
         aes(x = epi_pos, y = seq_pos, fill = log_attn)) +
    geom_tile() +
    scale_fill_viridis_c(
      name   = "log₁₀\n(attn)",
      limits = attn_lim,
      oob    = scales::squish,
      option = "viridis"
    ) +
    scale_x_continuous(name = "Epigenomic position (bp)",
                       breaks = c(0, 500, 1000)) +
    scale_y_continuous(name = "Sequence position (bp)",
                       breaks = c(0, 500, 1000)) +
    ggtitle(HEATMAP_TITLES[grp]) +
    theme_nar() +
    theme(
      aspect.ratio      = 1,
      plot.title        = element_text(size = 8, hjust = 0.5),
      legend.key.height = unit(0.8, "cm"),
      legend.key.width  = unit(0.25, "cm"),
      axis.text         = element_text(size = 8)
    )
}

heatmaps <- map(HEATMAP_GROUPS, \(g) make_heatmap(df_b, g))
heatmaps[[1]] <- heatmaps[[1]] +
  labs(tag = "C") +
  theme(plot.tag.position = c(0, 1),
        plot.tag          = element_text(face = "bold", size = 9, hjust = 0))

p_b <- wrap_plots(heatmaps, nrow = 1) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom",
        legend.key.height = unit(0.25, "cm"),
        legend.key.width  = unit(0.6,  "cm"))

# ── Panel B: row entropy + GGG density (all 3 marks) ────────────────────────

# r(row entropy, GGG density) ranges across the 4 G4+ groups
ENT_CORR <- c(
  "H3K4me3" = "r = -0.56 to -0.65",
  "H3K27ac" = "r = -0.27 to -0.41",
  "ATAC"    = "r = -0.01 to -0.13"
)

POS5_GROUPS <- c(
  "A549 G4+ promoter", "A549 G4+ enhancer",
  "HEK293T G4+ promoter", "HEK293T G4+ enhancer"
)

df_c <- read_csv("figures/data/fig4c.csv", show_col_types = FALSE) |>  # historical name
  filter(group %in% POS5_GROUPS) |>
  mutate(
    mark  = factor(mark, levels = MARK_LEVELS),
    group = factor(group, levels = names(ATTN_GROUP_COLORS))
  )

df_ggg <- read_csv("figures/data/fig4c_ggg.csv", show_col_types = FALSE)

ent_range_global <- range(df_c$row_entropy)
ggg_range        <- range(df_ggg$ggg_density)
df_ggg <- df_ggg |>
  mutate(ggg_scaled = (ggg_density - ggg_range[1]) /
           (ggg_range[2] - ggg_range[1]) *
           diff(ent_range_global) + ent_range_global[1])

make_entropy_panel <- function(mark_name, add_tag = FALSE, show_sec_axis = FALSE) {
  p <- ggplot() +
    geom_ribbon(
      data = df_ggg,
      aes(x = position_bp, ymin = ent_range_global[1], ymax = ggg_scaled),
      fill = "grey85", alpha = 0.8
    ) +
    geom_line(
      data = df_c |> filter(mark == mark_name),
      aes(x = position_bp, y = row_entropy, color = group),
      linewidth = 0.55
    ) +
    annotate("text", x = 990,
             y = ent_range_global[1] + 0.04 * diff(ent_range_global),
             label = ENT_CORR[mark_name],
             hjust = 1, vjust = 0, size = 2.3, color = "grey45",
             fontface = "italic") +
    scale_color_manual(values = ATTN_GROUP_COLORS[POS5_GROUPS], name = NULL) +
    scale_x_continuous(
      name   = "Position in window (bp)",
      limits = c(0, 1000), breaks = seq(0, 1000, 250)
    ) +
    scale_y_continuous(
      name     = "Row entropy (nats)",
      limits   = ent_range_global,
      expand   = expansion(mult = c(0, 0.05)),
      sec.axis = if (show_sec_axis) {
        sec_axis(
          ~ (. - ent_range_global[1]) / diff(ent_range_global) *
            diff(ggg_range) + ggg_range[1],
          name = "GGG density (shaded)"
        )
      } else {
        waiver()
      }
    ) +
    annotate("text", x = 5, y = Inf, label = mark_name,
             hjust = 0, vjust = 1.3, size = 2.8, fontface = "bold") +
    theme_nar() +
    theme(
      legend.position    = "none",
      axis.title.y.right = element_text(color = "grey60", size = 8)
    )

  if (add_tag) {
    p <- p + labs(tag = "B") +
      theme(plot.tag.position = c(0, 1),
            plot.tag          = element_text(face = "bold", size = 9, hjust = 0))
  }
  p
}

p_c_h3k4me3 <- make_entropy_panel("H3K4me3", add_tag = TRUE)
p_c_h3k27ac <- make_entropy_panel("H3K27ac")
p_c_atac    <- make_entropy_panel("ATAC", show_sec_axis = TRUE)

p_c <- (p_c_h3k4me3 + p_c_h3k27ac + p_c_atac) &
  theme(plot.margin = margin(2, 2, 2, 2))

# ── Assemble and save ────────────────────────────────────────────────────────

fig3 <- p_a / p_c / p_b +
  plot_layout(heights = c(1, 1, 1))

save_fig(fig3, "fig3", width_cm = 17.4, height_cm = 17)
