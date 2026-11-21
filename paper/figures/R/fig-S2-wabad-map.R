# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# Fig. S2 -- Where the 72 WABAD sites are and how Kinglet B compares with Perch 2.0 at each.
# a: world, colour = Kinglet B minus Perch 2.0 (pooled AUPRC over the site's windows and species list)
# b: Europe (27 sites; the rectangle in a), same colours
# Data: figure_data/wabad_per_site.csv (figures/build_figure_data.py)

source("figures/R/fig-theme.R")
library(dplyr)
library(patchwork)
theme_set(theme_paper(base_size = 8))

sites <- read.csv("figure_data/wabad_per_site.csv")
delta_lim <- 0.06
stopifnot(max(abs(sites$B - sites$teacher)) <= delta_lim)

world <- map_data("world") |> filter(region != "Antarctica")
europe <- list(x = c(-11, 36), y = c(33.5, 66))
stopifnot(sum(sites$lon > europe$x[1] & sites$lon < europe$x[2] &
                sites$lat > europe$y[1] & sites$lat < europe$y[2]) == 27)

site_map <- function(xlim, ylim) {
  ggplot() +
    geom_polygon(data = world, aes(long, lat, group = group), fill = "grey88", colour = NA) +
    geom_point(data = arrange(sites, abs(B - teacher)), aes(lon, lat, fill = B - teacher),
               shape = 21, size = 1.7, stroke = 0.2, colour = "black") +
    scale_fill_distiller(palette = "RdBu", direction = 1, limits = c(-delta_lim, delta_lim),
                         breaks = c(-0.06, -0.03, 0, 0.03, 0.06),
                         name = "Kinglet B − Perch 2.0 (AUPRC)") +
    coord_quickmap(xlim = xlim, ylim = ylim, expand = FALSE) +
    theme_paper(base_size = 8, style = "box") +
    theme(axis.title = element_blank(), axis.text = element_blank(), axis.ticks = element_blank())
}

p_world <- site_map(c(-165, 180), c(-55, 75)) +
  annotate("rect", xmin = europe$x[1], xmax = europe$x[2], ymin = europe$y[1], ymax = europe$y[2],
           fill = NA, colour = "black", linewidth = 0.25) +
  annotate_text_pt(europe$x[2] + 2, europe$y[2], "b", hjust = 0, vjust = 1, fontface = "bold")
p_europe <- site_map(europe$x, europe$y)

fig <- p_world + p_europe + plot_layout(widths = c(2.3, 1), guides = "collect") +
  plot_annotation(tag_levels = "a") &
  guides(fill = guide_colourbar(barwidth = unit(35, "mm"), barheight = unit(2, "mm"),
                                title.position = "left", title.vjust = 1)) &
  theme(legend.position = "bottom")
save_figure(fig, "figures/fig-S2-wabad-map.pdf", width_mm = W_2COL, height_mm = 75)
save_figure(fig, "figures/fig-S2-wabad-map.png", width_mm = W_2COL, height_mm = 75, dpi = 300)
