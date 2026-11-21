# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# Fig. S4 -- Species lists of Germany and Costa Rica, scored as in Fig. S3 on the benchmark recordings from
# outside each region (Germany: 496 listed species, 52 clusters, 361,547 windows; Costa Rica: 949 species,
# 53 clusters, 324,444 windows). a, c: false detections per window against recall of listed species;
# b, d: false detections at the recall of Perch 2.0 at p > 0.1, relative to Perch 2.0 (95% cluster bootstrap).
# BirdNET was not scored on these lists.
# Data: figure_data/openset_curves.csv, results/regionlists_v1.json (regionlists_v1)

source("figures/R/fig-theme.R")
library(jsonlite)
library(patchwork)
theme_set(theme_paper(base_size = 8))

curves <- read.csv("figure_data/openset_curves.csv")
res <- fromJSON("results/regionlists_v1.json")
titles <- c(DE = "Germany (496 species)", CR = "Costa Rica (949 species)")
rows <- lapply(names(titles), \(l) {
  p <- openset_panels(subset(curves, list == l), res[[l]]$models)
  (p$curves + labs(title = titles[[l]])) + p$rel + plot_layout(widths = c(1.5, 1))
})

fig <- rows[[1]] / rows[[2]] / guide_area() +
  plot_layout(heights = c(1, 1, 0.2), guides = "collect") + plot_annotation(tag_levels = "a")
save_figure(fig, "figures/fig-S4-region-lists.pdf", width_mm = W_2COL, height_mm = 150)
save_figure(fig, "figures/fig-S4-region-lists.png", width_mm = W_2COL, height_mm = 150, dpi = 300)
