# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# Fig. S1 -- Kinglet tracks Perch 2.0 site by site and species by species.
# a: per-site Kinglet B vs Perch 2.0 (pooled AUPRC over the site's windows and species list; 72 WABAD sites)
# b: per-site difference to Perch 2.0 for the three sizes
# c: per-species AP of each Kinglet size vs Perch 2.0, over all species-benchmark pairs of WABAD and the eight
#    BirdSet test sets
# The map of the sites is Fig. S2 (figures/R/fig-S2-wabad-map.R).
# Data: figure_data/wabad_per_site.csv, figure_data/species_ap.csv (figures/build_figure_data.py)

source("figures/R/fig-theme.R")
library(dplyr)
library(tidyr)
library(patchwork)
theme_set(theme_paper(base_size = 8))
set.seed(1)  # jitter in panel b

sites <- read.csv("figure_data/wabad_per_site.csv")
tiers <- c("B", "S", "XS")

# ---- a: site-level agreement -------------------------------------------------------------
p_a <- ggplot(sites, aes(teacher, B)) +
  geom_abline(linewidth = 0.25, linetype = "dashed") +
  geom_point(colour = MODEL_COLOURS[["B"]], shape = MODEL_SHAPES[["B"]], size = 1.1, alpha = 0.8) +
  annotate("text", x = 0.12, y = 0.86, hjust = 0, vjust = 1, size = pt_to_mm(7),
           label = sprintf("72 sites\nr = %.3f", cor(sites$teacher, sites$B))) +
  coord_equal(xlim = c(0.1, 0.9), ylim = c(0.1, 0.9)) +
  labs(x = "Perch 2.0 (site AUPRC)", y = "Kinglet B (site AUPRC)")

# ---- b: difference to the teacher by size --------------------------------------------------
delta <- sites |>
  pivot_longer(all_of(tiers), names_to = "tier", values_to = "auprc") |>
  mutate(delta = auprc - teacher, tier = factor(tier, levels = tiers))
medians <- delta |> group_by(tier) |> summarise(delta = median(delta))

p_b <- ggplot(delta, aes(tier, delta, colour = tier, shape = tier)) +
  geom_hline(yintercept = 0, linewidth = 0.25, linetype = "dashed") +
  geom_jitter(width = 0.22, height = 0, size = 0.9, alpha = 0.7) +
  geom_crossbar(data = medians, aes(ymin = delta, ymax = delta), width = 0.6, linewidth = 0.3,
                colour = "black") +
  scale_colour_manual(values = MODEL_COLOURS, guide = "none") +
  scale_shape_manual(values = MODEL_SHAPES, guide = "none") +
  scale_x_discrete(labels = MODEL_LABELS) +
  labs(x = NULL, y = "Difference to Perch 2.0 (site AUPRC)")

# ---- c: species-level agreement --------------------------------------------------------------
species <- read.csv("figure_data/species_ap.csv") |>
  pivot_longer(all_of(tiers), names_to = "tier", values_to = "ap") |>
  mutate(tier = factor(tier, levels = tiers))
r <- species |> group_by(tier) |>
  summarise(label = sprintf("%s species-benchmark pairs\nr = %.3f", format(n(), big.mark = ","), cor(ap, teacher)))

p_c <- ggplot(species, aes(teacher, ap, colour = tier)) +
  geom_abline(linewidth = 0.25, linetype = "dashed") +
  geom_point(size = 0.5, alpha = 0.5, stroke = 0) +
  geom_text(data = r, aes(0.02, 0.98, label = label), hjust = 0, vjust = 1, size = pt_to_mm(7),
            colour = "black", lineheight = 0.95) +
  facet_wrap(~ tier, labeller = as_labeller(MODEL_LABELS)) +
  scale_colour_manual(values = MODEL_COLOURS, guide = "none") +
  scale_x_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1")) +
  scale_y_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1")) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "Perch 2.0 (species AP)", y = "Kinglet (species AP)") +
  theme(strip.text = element_text(face = "plain", hjust = 0.5), panel.spacing = unit(4, "mm"))

fig <- (p_a + p_b + plot_layout(widths = c(1, 1.2))) / p_c +
  plot_layout(heights = c(1, 0.85)) + plot_annotation(tag_levels = "a")
save_figure(fig, "figures/fig-S1-sites-species.pdf", width_mm = W_2COL, height_mm = 132)
save_figure(fig, "figures/fig-S1-sites-species.png", width_mm = W_2COL, height_mm = 132, dpi = 300)
