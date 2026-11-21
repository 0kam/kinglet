# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# Fig. 4 -- Kinglet XS on a microcontroller.
# a: board photograph (figures/photo/board_grove_vision_ai_v2.jpg, the author's photograph; ruler in cm) and signal path
# b: board power over four inference windows, with one inference enlarged and the level between inferences
#    (median of the 10-ms smoothed trace, as in table3_mcu) marked
# c: embedding readouts of the microcontroller path (int8 trunk, feature restored to the Perch
#    space on the host) next to the fp32 model, both relative to Perch 2.0
# Data: figure_data/mcu_power_trace.csv (figures/build_figure_data.py),
#       results/wabad_emb{probe,compat}_results.json (mcuemb_v2)

source("figures/R/fig-theme.R")
library(dplyr)
library(patchwork)
library(jsonlite)
theme_set(theme_paper(base_size = 8))

# ---- a: photograph and signal path ----------------------------------------------------------
photo <- jpeg::readJPEG("figures/photo/board_grove_vision_ai_v2.jpg")
stopifnot(dim(photo)[1] == 1268, dim(photo)[2] == 1740)
photo <- photo[101:1268, 121:1700, ]  # drop the empty background above and at the sides
photo_w <- 28 * ncol(photo) / nrow(photo)  # width on the panel's mm-like scale at height 28
txt <- pt_to_mm(7)
nodes <- data.frame(
  x0 = c(41, 68, 96, 131, 96, 131), x1 = c(62, 90, 125, 179, 125, 179),
  y0 = c(15, 15, 15, 20, 1, 1), y1 = c(27, 27, 27, 27, 9, 9),
  label = c("Digital\nmicrophone", "Spectrogram\n(Cortex-M55)", "Kinglet XS, 8-bit\n(Ethos-U55 NPU)",
            "Compact output layer:\nscores of the listed species", "1,280-byte feature\nper window",
            "Later, on a computer:\nembedding in Perch 2.0 space"),
  kind = c("data", "student", "student", "out", "out", "out"))
p_a <- ggplot() +
  annotation_raster(photo, xmin = 1, xmax = 1 + photo_w, ymin = 1, ymax = 29) +
  geom_rect(data = nodes, aes(xmin = x0, xmax = x1, ymin = y0, ymax = y1, fill = kind), colour = "black",
            linewidth = 0.25) +
  geom_text(data = nodes, aes((x0 + x1) / 2, (y0 + y1) / 2, label = label), size = txt, lineheight = 0.95) +
  annotate("segment", x = c(62, 90, 125, 110.5, 125), xend = c(68, 96, 131, 110.5, 131),
           y = c(21, 21, 23.5, 15, 5), yend = c(21, 21, 23.5, 9, 5), linewidth = 0.25,
           arrow = arrow(length = unit(1.2, "mm"), type = "closed")) +
  annotate_text_pt(79, 12.5, "108 ms", size_pt = 6.5) +
  annotate_text_pt(110.5, 29, "366 ms", size_pt = 6.5) +
  scale_fill_manual(values = c(data = "#F0F0F0", student = "#FBE3B5", out = "#D6E9F5"), guide = "none") +
  coord_fixed(xlim = c(0, 180), ylim = c(0, 31), expand = FALSE) +
  theme_void() + theme(plot.tag = element_text(size = rel(1.1), face = "bold"))

# ---- b: power ------------------------------------------------------------------------
trace <- read.csv("figure_data/mcu_power_trace.csv")
between <- median(stats::filter(trace$mW, rep(1 / 10, 10)), na.rm = TRUE)  # level between inferences
stopifnot(between > 108 && between < 109)
burst <- trace$t_s[trace$mW > 135]                       # samples inside inference bursts
onsets <- burst[c(TRUE, diff(burst) > 1)]                # first sample of each burst
stopifnot(length(onsets) == 4, all(abs(diff(onsets) - 5.12) < 0.01))
zoom <- c(onsets[2] - 0.15, onsets[2] + 0.65)
y_lim <- c(72, 190)
stopifnot(min(trace$mW) > y_lim[1], max(trace$mW) < y_lim[2])  # nothing is clipped

p_wave <- ggplot(trace, aes(t_s, mW)) +
  geom_line(linewidth = 0.2) +
  geom_hline(yintercept = between, linewidth = 0.35, linetype = "dashed", colour = "#0072B2") +
  annotate("text", x = 0.3, y = 85, hjust = 0, size = pt_to_mm(7), colour = "#0072B2",
           label = sprintf("Dashed: between inferences, %.1f mW", between)) +
  annotate("segment", x = onsets[2], xend = onsets[3], y = 180, yend = 180, linewidth = 0.25,
           arrow = arrow(ends = "both", length = unit(1, "mm"))) +
  annotate("text", x = mean(onsets[2:3]), y = 185, label = "5.12 s", size = pt_to_mm(7)) +
  annotate("text", x = 0.3, y = 78, hjust = 0, size = pt_to_mm(7),
           label = sprintf("Mean %.0f mW", mean(trace$mW))) +
  scale_x_continuous(breaks = seq(0, 15, 5), expand = c(0, 0), limits = c(0, 17.5)) +
  scale_y_continuous(limits = y_lim, expand = c(0, 0)) +
  labs(x = "Time (s)", y = "Board power (mW)")

p_zoom <- ggplot(filter(trace, t_s >= zoom[1], t_s <= zoom[2]), aes(t_s - onsets[2], mW)) +
  geom_line(linewidth = 0.2) +
  scale_x_continuous(breaks = c(0, 0.5), expand = c(0, 0)) +
  scale_y_continuous(limits = y_lim, expand = c(0, 0)) +
  labs(x = "Time (s)", y = NULL, title = "One inference") +
  theme(axis.text.y = element_blank(), plot.title = element_text(size = rel(0.9), hjust = 0.5))

# ---- c: int8 microcontroller path vs fp32, relative to the teacher --------------------------
res_dir <- "results"
probe <- fromJSON(file.path(res_dir, "wabad_embprobe_results.json"))
compat <- fromJSON(file.path(res_dir, "wabad_embcompat_results.json"))
paths <- c(XS = "32-bit float", XS_int8pub = "8-bit")  # the board's 8-bit model run on a host (mcuemb_v2)
readout <- function(setting, label, teacher, value) {
  data.frame(setting = setting, readout = label, path = unname(paths),
             ratio = sapply(names(paths), value) / teacher)
}
own <- "Within Kinglet"
cross <- "Across spaces"
rel <- bind_rows(
  readout(own, "Retrieval, 1 example", probe$teacher$retrieval_k1_mAP, \(m) probe[[m]]$retrieval_k1_mAP),
  readout(own, "Retrieval, 5 examples", probe$teacher$retrieval_k5_mAP, \(m) probe[[m]]$retrieval_k5_mAP),
  readout(own, "Linear classifier", probe$teacher$probe_cmAP, \(m) probe[[m]]$probe_cmAP),
  readout(cross, "Teacher query, Kinglet archive, 1 ex.", probe$teacher$retrieval_k1_mAP,
          \(m) compat[[m]]$xretrieval_t2s_k1_mAP),
  readout(cross, "Teacher query, Kinglet archive, 5 ex.", probe$teacher$retrieval_k5_mAP,
          \(m) compat[[m]]$xretrieval_t2s_k5_mAP),
  readout(cross, "Kinglet query, teacher archive, 1 ex.", probe$teacher$retrieval_k1_mAP,
          \(m) compat[[m]]$xretrieval_s2t_k1_mAP),
  readout(cross, "Kinglet query, teacher archive, 5 ex.", probe$teacher$retrieval_k5_mAP,
          \(m) compat[[m]]$xretrieval_s2t_k5_mAP),
  readout(cross, "Teacher classifier, no refitting", compat$teacher_probe_cmAP, \(m) compat[[m]]$xprobe_cmAP)
) |>
  mutate(setting = factor(setting, levels = c(own, cross)),
         readout = factor(readout, levels = rev(unique(readout))),
         path = factor(path, levels = unname(paths)))

p_c <- ggplot(rel, aes(100 * ratio, readout)) +
  geom_vline(xintercept = 100, linewidth = 0.25, linetype = "dashed") +
  geom_point(aes(shape = path), colour = MODEL_COLOURS[["XS"]], size = 1.8, stroke = 0.5) +
  facet_grid(setting ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_shape_manual(values = c(15, 0), name = NULL) +
  scale_x_continuous(limits = c(83, 101), breaks = seq(85, 100, 5)) +
  labs(x = "Performance relative to Perch 2.0 (%)", y = NULL) +
  theme(strip.placement = "outside", strip.text.y.left = element_text(angle = 90, hjust = 0.5, face = "plain"),
        legend.position = "top", legend.justification = "left")

fig <- p_a / (p_wave + p_zoom + p_c + plot_layout(widths = c(1.25, 0.45, 1.1))) +
  plot_layout(heights = c(0.5, 1)) + plot_annotation(tag_levels = list(c("a", "b", "", "c")))
save_figure(fig, "figures/fig-04-mcu.pdf", width_mm = W_2COL, height_mm = 110)
save_figure(fig, "figures/fig-04-mcu.png", width_mm = W_2COL, height_mm = 110, dpi = 300)
