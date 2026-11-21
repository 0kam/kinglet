# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# fig-theme.R -- shared helpers for the Kinglet paper figures (ggplot2).
# Adapted from scipaper-skills paper-writing/references/figure-recipes.R (commit 3482045);
# column widths set for Methods in Ecology and Evolution (Wiley: 80-180 mm).
# Each figure is one script: Rscript figures/R/fig-0N-*.R (run from paper/).

library(ggplot2)

W_1COL <- 80    # mm, single column
W_2COL <- 180   # mm, full width

# ---------------------------------------------------------------------------
# 2. theme_paper() -- base theme for manuscript figures.
#    base_size is in POINTS. Use 7-9 for 90 mm figures; never leave it at the
#    ggplot2 default of 11 or the cowplot default of 14.
#    style = "classic" -> axis lines, no grid (scatter / line plots)
#    style = "box"     -> panel border, no grid (maps, rasters, facets)
# ---------------------------------------------------------------------------

theme_paper <- function(base_size = 8,
                        base_family = "",
                        style = c("classic", "box"),
                        legend_position = "top") {
  style <- match.arg(style)
  lw <- 0.25  # mm; ~0.7 pt line weight for axes and ticks

  base <- if (style == "classic") {
    theme_classic(base_size = base_size, base_family = base_family)
  } else {
    theme_bw(base_size = base_size, base_family = base_family)
  }

  out <- base +
    theme(
      text             = element_text(colour = "black"),
      axis.text        = element_text(size = rel(0.9), colour = "black"),
      axis.title       = element_text(size = rel(1.0), colour = "black"),
      axis.ticks       = element_line(colour = "black", linewidth = lw),
      legend.position  = legend_position,
      legend.key.size  = unit(3, "mm"),
      legend.title     = element_text(size = rel(0.9)),
      legend.text      = element_text(size = rel(0.9)),
      legend.margin    = margin(0, 0, 0, 0),
      strip.background = element_blank(),
      strip.text       = element_text(size = rel(1.0), hjust = 0, face = "bold"),
      plot.title       = element_text(size = rel(1.0), face = "plain"),
      plot.margin      = margin(2, 2, 2, 2, "mm"),
      plot.tag         = element_text(size = rel(1.1), face = "bold")
    )

  if (style == "classic") {
    out <- out + theme(axis.line = element_line(colour = "black", linewidth = lw))
  } else {
    out <- out + theme(
      panel.border = element_rect(fill = NA, colour = "black", linewidth = lw),
      panel.grid   = element_blank()
    )
  }
  out
}

# ---------------------------------------------------------------------------
# 3. Okabe-Ito categorical palette (colour-vision-deficiency safe).
#    Values verified with grDevices::palette.colors(9, "Okabe-Ito").
#    Order below puts black last so that the first colours are the most distinct.
# ---------------------------------------------------------------------------

okabeito_colours <- c(
  orange         = "#E69F00",
  skyblue        = "#56B4E9",
  bluishgreen    = "#009E73",
  yellow         = "#F0E442",
  blue           = "#0072B2",
  vermillion     = "#D55E00",
  reddishpurple  = "#CC79A7",
  grey           = "#999999",
  black          = "#000000"
)

# Return n Okabe-Ito colours. Errors above 9 categories on purpose: more than
# nine categories cannot be distinguished by colour alone (see figure-rules.md).
okabeito_pal <- function(n = 9, order = NULL) {
  cols <- unname(okabeito_colours)
  if (!is.null(order)) cols <- cols[order]
  if (n > length(cols)) {
    stop("Okabe-Ito provides at most 9 colours; recode categories or encode ",
         "them with shape/linetype instead of colour.", call. = FALSE)
  }
  cols[seq_len(n)]
}

scale_colour_okabeito <- function(..., order = NULL, aesthetics = "colour") {
  pal <- function(n) okabeito_pal(n, order = order)
  # discrete_scale() dropped the mandatory `scale_name` argument in ggplot2 3.5.0.
  if (utils::packageVersion("ggplot2") >= "3.5.0") {
    discrete_scale(aesthetics = aesthetics, palette = pal, ...)
  } else {
    discrete_scale(aesthetics = aesthetics, scale_name = "okabeito",
                   palette = pal, ...)
  }
}
scale_color_okabeito <- scale_colour_okabeito

scale_fill_okabeito <- function(..., order = NULL) {
  scale_colour_okabeito(..., order = order, aesthetics = "fill")
}

# ---------------------------------------------------------------------------
# 4. Point-size helpers for geom_text() / annotate().
#    TRAP: the `size` aesthetic of geom_text()/geom_label() and `linewidth` are
#    in MILLIMETRES, not points. size = 5 renders at about 14.2 pt.
#    ggplot2 >= 3.5.0 accepts size.unit = "pt"; older versions need pt_to_mm().
# ---------------------------------------------------------------------------

PT_PER_MM <- 72.27 / 25.4   # 2.845276 -- this is ggplot2's internal .pt

pt_to_mm <- function(pt) pt / PT_PER_MM   # e.g. pt_to_mm(7) -> 2.46

has_size_unit <- function() {
  utils::packageVersion("ggplot2") >= "3.5.0"
}

# Version-safe geom_text() wrapper: size_pt is always in points.
geom_text_pt <- function(..., size_pt = 7) {
  if (has_size_unit()) {
    geom_text(..., size = size_pt, size.unit = "pt")
  } else {
    geom_text(..., size = pt_to_mm(size_pt))
  }
}

annotate_text_pt <- function(x, y, label, size_pt = 7, ...) {
  if (has_size_unit()) {
    annotate("text", x = x, y = y, label = label,
             size = size_pt, size.unit = "pt", ...)
  } else {
    annotate("text", x = x, y = y, label = label,
             size = pt_to_mm(size_pt), ...)
  }
}

# ---------------------------------------------------------------------------
# 5. save_figure() -- ggsave wrapper enforcing real-world size in mm.
#    .pdf  -> grDevices::cairo_pdf (embeds fonts; plain pdf() does not)
#    .tif  -> ragg::agg_tiff with LZW compression (agg default is "none")
#    .png  -> ragg::agg_png
#    dpi defaults: 1000 for line art, use 500 for combination art and 300 for
#    halftone (Elsevier). Vector PDF ignores dpi except for rasterised layers.
# ---------------------------------------------------------------------------

save_figure <- function(plot, filename,
                        width_mm, height_mm,
                        dpi = 1000,
                        bg = "white",
                        ...) {
  ext <- tolower(tools::file_ext(filename))
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)

  if (ext == "pdf") {
    ggsave(filename, plot, device = grDevices::cairo_pdf,
           width = width_mm, height = height_mm, units = "mm",
           bg = bg, ...)
  } else if (ext %in% c("tif", "tiff")) {
    if (!requireNamespace("ragg", quietly = TRUE)) {
      stop("Package 'ragg' is required for TIFF output.", call. = FALSE)
    }
    ggsave(filename, plot, device = ragg::agg_tiff,
           width = width_mm, height = height_mm, units = "mm",
           dpi = dpi, bg = bg, compression = "lzw", ...)
  } else if (ext == "png") {
    if (!requireNamespace("ragg", quietly = TRUE)) {
      stop("Package 'ragg' is required for PNG output.", call. = FALSE)
    }
    ggsave(filename, plot, device = ragg::agg_png,
           width = width_mm, height = height_mm, units = "mm",
           dpi = dpi, bg = bg, ...)
  } else {
    stop("Use .pdf (vector), .tif/.tiff or .png. JPEG must not be used for ",
         "line art or text.", call. = FALSE)
  }
  invisible(normalizePath(filename))
}


# Kinglet tiers and reference models: one colour + shape per model across all figures.
MODEL_LABELS <- c(teacher = "Perch 2.0", B = "Kinglet B", S = "Kinglet S", XS = "Kinglet XS",
                  birdnet_v3 = "BirdNET+ V3.0", birdnet_v24 = "BirdNET V2.4")
MODEL_COLOURS <- c(teacher = "#000000", B = "#0072B2", S = "#009E73", XS = "#E69F00",
                   birdnet_v3 = "#999999", birdnet_v24 = "#999999")
MODEL_SHAPES <- c(teacher = 18, B = 16, S = 17, XS = 15, birdnet_v3 = 1, birdnet_v24 = 2)

# False detections per window against recall of the listed species (Fig. 4, Fig. S2).
# curves: rows of figure_data/openset_curves.csv for one list; res: the list's "models" entry of
# results/opensetcurve_v1.json or regionlists_v1.json. Returns two panels: the curves, with each model marked at
# the teacher's recall at p > 0.1, and the false detections at that recall relative to the teacher (95% interval
# from the cluster bootstrap).
OPENSET_SERIES <- c(teacher = "Perch 2.0",
                    B_exact = "Kinglet B, full output layer", S_exact = "Kinglet S, full output layer",
                    XS_exact = "Kinglet XS, full output layer", B_pack = "Kinglet B, compact output layer",
                    S_pack = "Kinglet S, compact output layer", XS_pack = "Kinglet XS, compact output layer",
                    v24 = "BirdNET V2.4", v3 = "BirdNET+ V3.0")
OPENSET_COLOURS <- c(teacher = "#000000", B_exact = "#0072B2", S_exact = "#009E73", XS_exact = "#E69F00",
                     B_pack = "#0072B2", S_pack = "#009E73", XS_pack = "#E69F00", v24 = "#999999", v3 = "#999999")
OPENSET_LINETYPES <- c(teacher = "solid", B_exact = "solid", S_exact = "solid", XS_exact = "solid",
                       B_pack = "22", S_pack = "22", XS_pack = "22", v24 = "13", v3 = "1343")

openset_panels <- function(curves, res, xlim = c(5e-4, 2)) {
  series <- intersect(names(OPENSET_SERIES), unique(curves$model))
  stopifnot(setequal(series, names(res)))
  at <- lapply(res, \(m) m$`at_recall_teacher_recall@0.1`)
  teacher_recall <- res$teacher$`p>0.1`$recall
  stopifnot(abs(sapply(at, `[[`, "recall") - teacher_recall) < 1e-9)
  marks <- data.frame(model = factor(names(at), levels = series), fp_per_window = sapply(at, `[[`, "fp_per_window"))
  curves$model <- factor(curves$model, levels = series)
  p_curves <- ggplot(curves, aes(fp_per_window, recall, colour = model, linetype = model)) +
    geom_hline(yintercept = teacher_recall, linewidth = 0.25, linetype = "dotted", colour = "grey40") +
    geom_line(linewidth = 0.45) +
    geom_point(data = marks, aes(fp_per_window, teacher_recall), size = 1.2, show.legend = FALSE) +
    scale_colour_manual(values = OPENSET_COLOURS[series], labels = OPENSET_SERIES[series], name = NULL) +
    scale_linetype_manual(values = OPENSET_LINETYPES[series], labels = OPENSET_SERIES[series], name = NULL) +
    scale_x_log10(breaks = 10^(-4:1), labels = c("0.0001", "0.001", "0.01", "0.1", "1", "10")) +
    coord_cartesian(xlim = xlim, ylim = c(0, 0.9), expand = FALSE) +
    labs(x = "False detections per window", y = "Recall of listed species") +
    guides(colour = guide_legend(ncol = 3), linetype = guide_legend(ncol = 3)) +
    theme(legend.key.width = unit(7, "mm"))

  rel <- do.call(rbind, lapply(setdiff(series, "teacher"), \(m) {
    r <- res[[m]]$R_vs_teacher_at_teacher_recall
    data.frame(model = m, R = r$R, lo = r$ci95[1], hi = r$ci95[2])
  }))
  stopifnot(abs(rel$R - sapply(at[rel$model], `[[`, "fp_per_window") / at$teacher$fp_per_window) < 1e-6)
  rel$label <- factor(OPENSET_SERIES[rel$model], levels = rev(OPENSET_SERIES[rel$model]))
  p_rel <- ggplot(rel, aes(R, label, colour = model)) +
    geom_vline(xintercept = 1, linewidth = 0.25, linetype = "dashed") +
    geom_linerange(aes(xmin = lo, xmax = hi), linewidth = 0.35) +
    geom_point(aes(shape = grepl("pack", model)), size = 1.4, fill = "white", stroke = 0.6) +
    scale_colour_manual(values = OPENSET_COLOURS, guide = "none") +
    scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 21), guide = "none") +
    scale_x_log10(breaks = c(0.1, 0.25, 0.5, 1, 2, 4), labels = c("0.1", "0.25", "0.5", "1", "2", "4")) +
    expand_limits(x = c(0.5, 2)) +
    labs(x = "False detections relative to\nPerch 2.0 at equal recall", y = NULL)
  list(curves = p_curves, rel = p_rel)
}
