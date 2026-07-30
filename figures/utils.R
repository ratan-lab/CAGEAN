library(tidyverse)
library(patchwork)
library(magick)

# Okabe-Ito colorblind-safe palette
OI <- c(
  orange = "#E69F00",
  sky    = "#56B4E9",
  green  = "#009E73",
  yellow = "#F0E442",
  blue   = "#0072B2",
  vermil = "#D55E00",
  purple = "#CC79A7",
  black  = "#000000",
  gray   = "#999999"
)

MARK_COLORS <- c(
  H3K4me3 = unname(OI["blue"]),
  H3K27ac = unname(OI["green"]),
  ATAC    = unname(OI["vermil"])
)

MARK_LEVELS <- c("H3K4me3", "H3K27ac", "ATAC")

CELL_COLORS <- c(
  "A549"    = unname(OI["blue"]),
  "HEK293T" = unname(OI["orange"]),
  "H9 ESC"  = unname(OI["green"])
)

CELL_LEVELS <- c("A549", "HEK293T", "H9 ESC")

STRATUM_COLORS <- c(
  Promoter = unname(OI["blue"]),
  Enhancer = unname(OI["orange"]),
  Inactive = unname(OI["gray"])
)

MODEL_COLORS <- c(
  "epiG4NN"              = unname(OI["purple"]),
  "CAGEAN"               = unname(OI["blue"]),
  "Seq-only Transformer" = unname(OI["sky"]),
  "Seq-only CNN"         = unname(OI["gray"])
)

ATTN_GROUP_COLORS <- c(
  "A549 G4+ all"          = unname(OI["black"]),
  "A549 G4+ promoter"     = unname(OI["blue"]),
  "A549 G4+ enhancer"     = unname(OI["orange"]),
  "HEK293T G4+ promoter"  = unname(OI["sky"]),
  "HEK293T G4+ enhancer"  = unname(OI["purple"]),
  "A549 G4-"              = unname(OI["gray"])
)

ATTN_GROUP_LTYS <- c(
  "A549 G4+ all"          = "solid",
  "A549 G4+ promoter"     = "solid",
  "A549 G4+ enhancer"     = "solid",
  "HEK293T G4+ promoter"  = "dashed",
  "HEK293T G4+ enhancer"  = "dashed",
  "A549 G4-"              = "dotted"
)

svg_to_panel <- function(path, tag = NULL, width_px = 1600) {
  img  <- magick::image_read_svg(path, width = width_px)
  info <- magick::image_info(img)
  p <- ggplot() +
    annotation_raster(as.raster(img),
                      xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf) +
    coord_fixed(ratio = info$height / info$width) +
    theme_void()
  if (!is.null(tag)) p <- p + labs(tag = tag)
  p
}

# NAR Genomics base theme (8pt minimum font)
theme_nar <- function(base_size = 8) {
  theme_classic(base_size = base_size) +
    theme(
      strip.background   = element_blank(),
      strip.text         = element_text(face = "bold", size = base_size),
      axis.line          = element_line(linewidth = 0.35),
      axis.ticks         = element_line(linewidth = 0.35),
      axis.title         = element_text(size = base_size),
      axis.text          = element_text(size = base_size),
      legend.key.size    = unit(0.3, "cm"),
      legend.title       = element_text(size = base_size),
      legend.text        = element_text(size = base_size),
      legend.background  = element_blank(),
      plot.title         = element_text(face = "bold", size = base_size),
      plot.tag           = element_text(face = "bold", size = base_size + 1),
      panel.grid.major.y = element_line(color = "grey90", linewidth = 0.25),
      panel.grid.major.x = element_blank()
    )
}

# Save TIFF (600 dpi) and PDF (vector)
save_fig <- function(plot, name, width_cm, height_cm) {
  dir.create("figures", showWarnings = FALSE)
  tiff_path <- file.path("figures", paste0(name, ".tiff"))
  pdf_path  <- file.path("figures", paste0(name, ".pdf"))
  ggsave(tiff_path, plot, width = width_cm, height = height_cm,
         units = "cm", dpi = 600, compression = "lzw", bg = "white")
  ggsave(pdf_path,  plot, width = width_cm, height = height_cm,
         units = "cm", device = pdf, bg = "white")
  message("Saved: ", tiff_path, " and ", pdf_path)
}
