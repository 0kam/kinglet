# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# Fig. S3 -- With the Japanese species list, Kinglet makes about as many false detections as Perch 2.0 at the
# same recall, on recordings where 94% of the annotated occurrences are of species not on the list.
# a: false detections (a listed species above the threshold in a window where it is not annotated) per window
#    against recall of the annotated occurrences of listed species, over all thresholds, on the eight benchmarks
#    (381,911 windows). Perch 2.0 reads the listed species from its softmax over all classes; Kinglet with the
#    full output layer (softmax over all 14,795 classes, as on computers) and with the compact output layer
#    (as on the microcontroller); BirdNET confidence scores for reference. Dotted line: recall of Perch 2.0 at
#    p > 0.1; points: each model at that recall.
# b: false detections per window at that recall relative to Perch 2.0, with 95% intervals from a bootstrap over
#    clusters (72 WABAD sites and 7 BirdSet sets)
# Data: figure_data/openset_curves.csv, results/opensetcurve_v1.json (opensetcurve_v1),
#       results/birdnetv3rel_v2.json (BirdNET+ V3.0 final release)

source("figures/R/fig-theme.R")
library(jsonlite)
library(patchwork)
theme_set(theme_paper(base_size = 8))

curves <- subset(read.csv("figure_data/openset_curves.csv"), list == "JP")
res <- fromJSON("results/opensetcurve_v1.json")$JP$models
# BirdNET+ V3.0: final release, each 5-s window as one input, species matched by ebird_code (birdnetv3rel_v2)
res$v3 <- fromJSON("results/birdnetv3rel_v2.json")$openset_JP$v3
p <- openset_panels(curves, res)
p$curves <- p$curves +
  annotate_text_pt(1.8, res$teacher$`p>0.1`$recall - 0.035, "Recall of Perch 2.0 at threshold 0.1", size_pt = 6,
                   hjust = 1, colour = "grey30")

fig <- (p$curves + p$rel + plot_layout(widths = c(1.5, 1))) / guide_area() +
  plot_layout(heights = c(1, 0.17), guides = "collect") + plot_annotation(tag_levels = "a")
save_figure(fig, "figures/fig-S3-species-lists.pdf", width_mm = W_2COL, height_mm = 85)
save_figure(fig, "figures/fig-S3-species-lists.png", width_mm = W_2COL, height_mm = 85, dpi = 300)
