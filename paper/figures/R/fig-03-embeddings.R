# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# Fig. 3 -- Kinglet embeddings live in the Perch 2.0 space, for birds and for other taxa.
# a: per-window cosine similarity between teacher and student embeddings (WABAD, 60,158 windows)
# b: WABAD readouts relative to the teacher, in the student's own space (re-fit) and across spaces (assets built
#    on Perch embeddings applied unchanged); the teacher's absolute value of each readout printed on the right
# c: the same readouts on four non-bird sets (no training audio of these taxa): mean of the three own-space
#    ratios, two cross-space readouts, and AP of the species scores (full output layer) for the 36 frog species
#    of AnuraSet that have a class
# Data: figure_data/embcompat_cos_hist.csv (figures/build_figure_data.py),
#       results/wabad_emb{probe,compat}_results.json, nonbird_v1.json

source("figures/R/fig-theme.R")
library(dplyr)
library(patchwork)
library(jsonlite)
theme_set(theme_paper(base_size = 8))

tiers <- c("B", "S", "XS")
res_dir <- "results"
probe <- fromJSON(file.path(res_dir, "wabad_embprobe_results.json"))
compat <- fromJSON(file.path(res_dir, "wabad_embcompat_results.json"))
nonbird <- fromJSON(file.path(res_dir, "nonbird_v1.json"))
dodge <- c(B = 0.25, S = 0, XS = -0.25)  # vertical offsets of the three sizes within a row

# ---- a: field alignment -------------------------------------------------------------
cos_min <- 0.5
hist <- read.csv("figure_data/embcompat_cos_hist.csv") |>
  group_by(tier) |>
  mutate(density = n / sum(n) / 0.01, cos_mid = cos_lo + 0.005) |>
  ungroup()
stopifnot(sum(hist$n[hist$cos_lo < cos_min]) / sum(hist$n) < 1e-3)  # axis cut hides < 0.1% of windows
hist <- filter(hist, cos_lo >= cos_min)
medians <- data.frame(tier = tiers, y = c(16, 14.3, 12.6),
                      median = sapply(tiers, \(t) compat[[t]]$field_cos_median))

p_a <- ggplot(hist, aes(cos_mid, density, colour = tier)) +
  geom_line(linewidth = 0.5) +
  annotate("text", x = 0.52, y = 17.7, label = "Median", hjust = 0, size = pt_to_mm(7)) +
  geom_text(data = medians, aes(0.52, y, label = sprintf("%s  %.2f", MODEL_LABELS[tier], median)),
            size = pt_to_mm(7), hjust = 0) +
  scale_colour_manual(values = MODEL_COLOURS, guide = "none") +
  scale_x_continuous(limits = c(cos_min, 1), breaks = seq(0.5, 1, 0.1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 18.5), expand = c(0, 0)) +
  labs(x = "Cosine similarity to Perch 2.0", y = "Density of windows") +
  theme(plot.margin = margin(2, 5, 2, 2, "mm"))

# ---- b: WABAD readouts relative to the teacher ----------------------------------------
readout <- function(setting, label, teacher, student) {
  data.frame(setting = setting, readout = label, tier = tiers, teacher = teacher,
             ratio = sapply(tiers, student) / teacher)
}
own <- "Within Kinglet"      # index or probe re-fit on Kinglet embeddings
cross <- "Across spaces"  # assets built on Perch embeddings, applied unchanged
rel <- bind_rows(
  readout(own, "Retrieval, 1 example", probe$teacher$retrieval_k1_mAP, \(t) probe[[t]]$retrieval_k1_mAP),
  readout(own, "Retrieval, 5 examples", probe$teacher$retrieval_k5_mAP, \(t) probe[[t]]$retrieval_k5_mAP),
  readout(own, "Linear classifier", probe$teacher$probe_cmAP, \(t) probe[[t]]$probe_cmAP),
  readout(cross, "Teacher query, Kinglet archive, 1 ex.", probe$teacher$retrieval_k1_mAP,
          \(t) compat[[t]]$xretrieval_t2s_k1_mAP),
  readout(cross, "Teacher query, Kinglet archive, 5 ex.", probe$teacher$retrieval_k5_mAP,
          \(t) compat[[t]]$xretrieval_t2s_k5_mAP),
  readout(cross, "Kinglet query, teacher archive, 1 ex.", probe$teacher$retrieval_k1_mAP,
          \(t) compat[[t]]$xretrieval_s2t_k1_mAP),
  readout(cross, "Kinglet query, teacher archive, 5 ex.", probe$teacher$retrieval_k5_mAP,
          \(t) compat[[t]]$xretrieval_s2t_k5_mAP),
  readout(cross, "Teacher classifier, no refitting", compat$teacher_probe_cmAP, \(t) compat[[t]]$xprobe_cmAP)
) |>
  mutate(setting = factor(setting, levels = c(own, cross)),
         readout = factor(readout, levels = rev(unique(readout))))
stopifnot(compat$teacher_probe_cmAP == probe$teacher$probe_cmAP)
teacher_abs <- distinct(rel, setting, readout, teacher)
x_abs <- 106  # column of the teacher's absolute values

p_b <- ggplot(rel, aes(100 * ratio, readout)) +
  geom_vline(xintercept = 100, linewidth = 0.25, linetype = "dashed") +
  geom_line(aes(group = readout), colour = "grey70", linewidth = 0.3) +
  geom_point(aes(colour = tier, shape = tier), size = 1.6) +
  geom_text(data = teacher_abs, aes(x_abs, readout, label = sprintf("%.3f", teacher)), size = pt_to_mm(7)) +
  geom_text(data = filter(teacher_abs, readout == levels(readout)[nlevels(readout)]),
            aes(x_abs, readout, label = "Perch 2.0"), size = pt_to_mm(7), vjust = -1.4) +
  facet_grid(setting ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_colour_manual(values = MODEL_COLOURS, labels = MODEL_LABELS, name = NULL) +
  scale_shape_manual(values = MODEL_SHAPES, labels = MODEL_LABELS, name = NULL) +
  scale_x_continuous(breaks = seq(85, 100, 5)) +
  coord_cartesian(xlim = c(84, 103), clip = "off") +
  labs(x = "Performance relative to Perch 2.0 (%)", y = NULL) +
  theme(strip.placement = "outside", strip.text.y.left = element_text(angle = 90, hjust = 0.5, face = "plain"),
        legend.position = "top", legend.justification = "left", plot.margin = margin(2, 12, 2, 2, "mm"))

# ---- c: non-bird sets ------------------------------------------------------------------
sets <- c(watkins = "Watkins\nmarine mammals", anuraset = "AnuraSet\nfrogs",
          humbugdb = "HumBugDB\nmosquitoes", reefset = "ReefSet\ncoral reefs")
readouts <- c(within = "Within Kinglet\n(mean of the 3 analyses)",
              t2s = "Across spaces\nteacher query,\nKinglet archive, 1 ex.",
              xprobe = "Across spaces\nteacher classifier,\nno refitting",
              head = "Species scores\n(36 frog species)")
nb <- bind_rows(lapply(names(sets), \(s) {
  e <- nonbird[[s]]$embedding
  bind_rows(lapply(tiers, \(t) {
    r <- e[[paste0("ratio_", t)]]
    within <- mean(c(e[[t]]$retrieval_k1_mAP / e$teacher$retrieval_k1_mAP,
                     e[[t]]$retrieval_k5_mAP / e$teacher$retrieval_k5_mAP, e[[t]]$probe_cmAP / e$teacher$probe_cmAP))
    stopifnot(abs(within - r$mean_within) < 1e-3)
    out <- data.frame(set = s, tier = t, readout = c("within", "t2s", "xprobe"),
                      ratio = c(within, e[[t]]$xretrieval_t2s_k1_mAP / e$teacher$retrieval_k1_mAP,
                                e[[t]]$xprobe_cmAP / e$teacher$probe_cmAP))
    if (s == "anuraset") {  # head readout: student softmax AP over teacher logit AP (nonbird_head_v1 primary)
      h <- nonbird[[s]]$head
      stopifnot(h$n_scored == 36)
      out <- rbind(out, data.frame(set = s, tier = t, readout = "head",
                                   ratio = h$macro_AP[[t]] / h$macro_AP$teacher_logit))
    }
    out
  }))
})) |>
  mutate(set = factor(set, levels = rev(names(sets))), readout = factor(readout, levels = names(readouts)),
         y = as.numeric(set) + dodge[tier])

p_c <- ggplot(nb, aes(100 * ratio, y)) +
  geom_vline(xintercept = 100, linewidth = 0.25, linetype = "dashed") +
  geom_point(aes(colour = tier, shape = tier), size = 1.4) +
  facet_grid(. ~ readout, scales = "free_x", labeller = as_labeller(readouts)) +
  scale_colour_manual(values = MODEL_COLOURS, guide = "none") +
  scale_shape_manual(values = MODEL_SHAPES, guide = "none") +
  scale_x_continuous(breaks = scales::breaks_extended(n = 4), expand = expansion(mult = 0.15)) +
  scale_y_continuous(breaks = seq_along(sets), labels = rev(sets), expand = expansion(add = 0.5)) +
  labs(x = "Performance relative to Perch 2.0 (%)", y = NULL) +
  theme(axis.ticks.y = element_blank(), strip.text = element_text(face = "plain", hjust = 0.5, size = rel(0.9)),
        panel.spacing = unit(3, "mm"))

fig <- (free(p_a) + p_b + plot_layout(widths = c(0.85, 1.3))) / p_c + plot_layout(heights = c(1, 0.8)) +
  plot_annotation(tag_levels = "a")
save_figure(fig, "figures/fig-03-embeddings.pdf", width_mm = W_2COL, height_mm = 140)
save_figure(fig, "figures/fig-03-embeddings.png", width_mm = W_2COL, height_mm = 140, dpi = 300)
