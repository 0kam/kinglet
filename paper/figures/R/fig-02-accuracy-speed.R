# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# Fig. 2 (key figure) -- Kinglet keeps most of Perch 2.0's accuracy at many times its CPU throughput.
# a: accuracy (WABAD cmAP) against end-to-end CPU throughput (decoding included) for Kinglet sizes
#    and reference models
# b: Kinglet cmAP relative to Perch 2.0 on WABAD and the eight BirdSet test sets, with 95% intervals
#    from a paired bootstrap over species; the two sets that also served as development gates
#    (POW, HSN) are listed last, marked and drawn lighter
# Data: figure_data/benchmarks.csv, figure_data/speed.csv (figures/build_figure_data.py),
#       results/revision_stats.json ("bootstrap" -> "species")
# Throughput: figure_data/speed.csv needs the columns `model` (teacher, B, S, XS, birdnet_v3, birdnet_v24)
#   `machine`, `threads` and `xrt_e2e` (end-to-end throughput, x real time, from the speed kit); panel a
#   shows the four-thread rows of SPEED_MACHINE.

source("figures/R/fig-theme.R")
library(dplyr)
library(patchwork)
library(jsonlite)
theme_set(theme_paper(base_size = 8))

SPEED_MACHINE <- "i7-8565U"  # reference laptop (Intel Core i7-8565U), 4 threads

bench <- read.csv("figure_data/benchmarks.csv")
tiers <- c("B", "S", "XS")
speed <- read.csv("figure_data/speed.csv")
speed <- filter(speed, machine == SPEED_MACHINE, threads == 4)
stopifnot(setequal(speed$model, names(MODEL_LABELS)), !anyDuplicated(speed$model))

# ---- a: accuracy vs throughput ---------------------------------------------------------
pareto <- bench |>
  filter(benchmark == "WABAD") |>
  left_join(speed, by = "model") |>
  mutate(label = MODEL_LABELS[model],
         # labels: Perch above, XS below, B and S to their left, BirdNET to its right
         # (each label on the far side from the neighbouring size's symbol)
         side = case_when(grepl("birdnet", model) ~ "right", model %in% c("B", "S") ~ "left",
                          model == "XS" ~ "below", TRUE ~ "above"),
         x_lab = xrt_e2e * 10^c(above = 0, below = 0, left = -0.04, right = 0.04)[side],
         y_lab = cmAP + c(above = 0.017, below = -0.015, left = 0, right = 0)[side],
         hjust = c(above = 0.5, below = 0.5, right = 0, left = 1)[side])
teacher_cmap <- pareto$cmAP[pareto$model == "teacher"]
x_lim <- range(pareto$xrt_e2e) * c(0.75, 1.5)  # follows the speed values, room for the labels

p_a <- ggplot(pareto, aes(xrt_e2e, cmAP)) +
  geom_hline(yintercept = teacher_cmap, linewidth = 0.25, linetype = "dashed") +
  geom_point(aes(colour = model, shape = model), size = 2, stroke = 0.5) +
  geom_text(aes(x = x_lab, y = y_lab, label = label, hjust = hjust), size = pt_to_mm(7)) +
  scale_colour_manual(values = MODEL_COLOURS, guide = "none") +
  scale_shape_manual(values = MODEL_SHAPES, guide = "none") +
  scale_x_log10(limits = x_lim, breaks = c(5, 10, 20, 50, 100, 200, 500, 1000, 2000),
                sec.axis = sec_axis(~ 1000 / ., breaks = c(200, 100, 50, 20, 10, 5, 2, 1, 0.5),
                                    name = "Time to analyse 1,000 h of audio (h)")) +
  scale_y_continuous(limits = c(0.18, 0.44), breaks = seq(0.2, 0.4, 0.05)) +
  labs(x = "End-to-end CPU throughput (× real time, 4 threads)", y = "WABAD cmAP")

# ---- b: relative accuracy across benchmarks, with paired bootstrap intervals ------------------
boot <- fromJSON("results/revision_stats.json")$bootstrap$species
rel <- bench |>
  group_by(benchmark) |>
  mutate(ratio = cmAP / cmAP[model == "teacher"]) |>
  ungroup() |>
  filter(model %in% tiers) |>
  rowwise() |>
  mutate(boot_ratio = boot[[paste(benchmark, model)]]$ratio,
         lo = boot[[paste(benchmark, model)]]$ratio_ci[1],
         hi = boot[[paste(benchmark, model)]]$ratio_ci[2]) |>
  ungroup()
stopifnot(abs(rel$boot_ratio - rel$ratio) < 1e-3)  # the intervals belong to these very ratios
rel <- rel |>
  mutate(dev = as.logical(development_set),
         label = paste0(ifelse(benchmark == "WABAD", "", "BirdSet "), benchmark, ifelse(dev, " †", "")),
         # WABAD first, then BirdSet alphabetically; development sets at the bottom
         order = rank(paste(dev, benchmark != "WABAD", benchmark), ties.method = "min"),
         benchmark = factor(label, levels = rev(unique(label[order(order)]))),
         y = as.numeric(benchmark) + c(B = 0.25, S = 0, XS = -0.25)[model])
n_dev <- sum(tapply(rel$dev, rel$benchmark, any))

p_b <- ggplot(rel, aes(ratio, y)) +
  geom_vline(xintercept = 1, linewidth = 0.25, linetype = "dashed") +
  geom_hline(yintercept = n_dev + 0.5, linewidth = 0.25, linetype = "dotted") +
  geom_linerange(aes(xmin = lo, xmax = hi, colour = model, alpha = dev), linewidth = 0.35) +
  geom_point(aes(colour = model, shape = model, alpha = dev), size = 1.3) +
  scale_alpha_manual(values = c(`FALSE` = 1, `TRUE` = 0.45), guide = "none") +
  geom_text(data = filter(rel, benchmark == "WABAD"), aes(x = lo - 0.006, label = model, colour = model),
            size = pt_to_mm(6.5), hjust = 1) +
  scale_colour_manual(values = MODEL_COLOURS, guide = "none") +
  scale_shape_manual(values = MODEL_SHAPES, guide = "none") +
  scale_x_continuous(limits = c(0.72, 1.1), breaks = seq(0.75, 1.1, 0.05)) +
  scale_y_continuous(breaks = seq_along(levels(rel$benchmark)), labels = levels(rel$benchmark),
                     expand = expansion(add = 0.5)) +
  labs(x = "cmAP relative to Perch 2.0", y = NULL) +
  theme(axis.ticks.y = element_blank())

fig <- p_a + p_b + plot_layout(widths = c(1.3, 1)) + plot_annotation(tag_levels = "a")
save_figure(fig, "figures/fig-02-accuracy-speed.pdf", width_mm = W_2COL, height_mm = 85)
save_figure(fig, "figures/fig-02-accuracy-speed.png", width_mm = W_2COL, height_mm = 85, dpi = 300)
