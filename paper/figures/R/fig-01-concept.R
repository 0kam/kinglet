# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# Fig. 1 -- What Kinglet is: one distillation from public recordings and the teacher's outputs; a species list
# chooses the output (computers: the full output layer, all 14,795 classes; the microcontroller: a compact output
# layer, because the full one does not fit); the same network on computers and a microcontroller. Every node of
# the data-flow graph in figure-plan.md (N1, N3-N6, N8-N11; N7 = the compact output layer).
# Data: figure_data/fig1_spectrogram.csv (a real WABAD window; build_figure_data.py)

source("figures/R/fig-theme.R")
theme_set(theme_paper(base_size = 8))

W <- 180; H <- 104                                   # canvas in mm
FILL <- c(data = "#F0F0F0", teacher = "#D9D9D9", student = "#FBE3B5", out = "#D6E9F5", note = NA)
txt <- pt_to_mm(7)

boxes <- list(); arrows <- list(); labels <- list()
box <- function(x0, x1, y0, y1, label, kind = "data", face = "plain") {
  boxes[[length(boxes) + 1]] <<- data.frame(x0, x1, y0, y1, label, kind, face)
}
arrow_ <- function(x, y, xend, yend, head = TRUE) arrows[[length(arrows) + 1]] <<- data.frame(x, y, xend, yend, head)
label <- function(x, y, label, hjust = 0, face = "plain") {
  labels[[length(labels) + 1]] <<- data.frame(x, y, label, hjust, face)
}

# ---- 1. distil once (top band) ---------------------------------------------------------
label(1, 101, "1  Distil once", face = "bold")
box(1, 37, 72, 94, "Public focal recordings\n428,482 recordings\n9,751 species")
box(49, 89, 86, 96, "Perch 2.0 (teacher)", "teacher")
box(49, 89, 70, 80, "Kinglet (B, S or XS)", "student")
box(99, 139, 91.5, 96.5, "Embedding", "teacher")
box(99, 139, 85.5, 90.5, "Probabilities of 14,795 classes", "teacher")
box(99, 139, 75.5, 80.5, "Embedding layer", "student")
box(99, 139, 69.5, 74.5, "Full output layer", "student")
box(149, 179, 74, 92, "Loss:\nmatch the teacher", "note")
arrow_(37, 85, 49, 91); arrow_(37, 81, 49, 75)
arrow_(89, 91, 99, 94); arrow_(89, 91, 99, 88); arrow_(89, 75, 99, 78); arrow_(89, 75, 99, 72)
arrow_(139, 94, 149, 88); arrow_(139, 88, 149, 85); arrow_(139, 78, 149, 81); arrow_(139, 72, 149, 78)

# ---- 2. choose a species list (middle band) -----------------------------------------------
label(1, 63, "2  Choose a species list, without retraining", face = "bold")
box(1, 37, 40, 57, "Species list\n(e.g. 317 species\nof Japan)")
box(49, 179, 50.5, 58.5, "Computer: full output layer\n(all classes are scored; the listed species are read out)", "out")
box(49, 179, 38.5, 48.5, "Microcontroller: compact output layer\n(the listed species + one output for all other classes)", "out")
arrow_(37, 48.5, 49, 54.5); arrow_(37, 48.5, 49, 43.5); arrow_(119, 69.5, 119, 58.5)

# ---- 3. run on a computer or a microcontroller (bottom band) ------------------------------
label(1, 32.5, "3  Run on a computer or a microcontroller", face = "bold")
label(19, 27.5, "Audio, 5-s windows", hjust = 0.5)
box(45, 63, 8, 22, "Kinglet", "student")
box(73, 115, 23, 30, "Full output layer\n(computer)", "student")
box(73, 115, 13, 20, "Compact output layer\n(microcontroller)", "student")
box(73, 115, 1, 11, "Embedding layer", "student")
box(123, 179, 13, 30, "Scores of the listed species", "out")
box(123, 179, 1, 11, "Embedding in the Perch 2.0 space", "out")
arrow_(37, 15, 45, 15)
arrow_(63, 15, 73, 26.5); arrow_(63, 15, 73, 16.5); arrow_(63, 15, 73, 6)
arrow_(115, 26.5, 123, 26.5); arrow_(115, 16.5, 123, 16.5); arrow_(115, 6, 123, 6)

boxes <- do.call(rbind, boxes); arrows <- do.call(rbind, arrows); labels <- do.call(rbind, labels)
spec <- as.matrix(read.csv("figure_data/fig1_spectrogram.csv"))
spec <- spec[nrow(spec):1, ] / 255                    # high frequencies on top
ras <- as.raster(matrix(grDevices::grey(1 - spec), nrow = nrow(spec)))

fig <- ggplot() +
  annotate("segment", x = 0, xend = W, y = c(66, 35), yend = c(66, 35), linewidth = 0.2, colour = "grey60",
           linetype = "dotted") +
  geom_rect(data = boxes, aes(xmin = x0, xmax = x1, ymin = y0, ymax = y1, fill = kind),
            colour = "black", linewidth = 0.25) +
  geom_text(data = boxes, aes((x0 + x1) / 2, (y0 + y1) / 2, label = label), size = txt, lineheight = 0.95) +
  geom_text(data = labels, aes(x, y, label = label, hjust = hjust, fontface = face), size = txt, lineheight = 0.95) +
  geom_segment(data = subset(arrows, head), aes(x, y, xend = xend, yend = yend), linewidth = 0.25,
               arrow = arrow(length = unit(1.2, "mm"), type = "closed")) +
  geom_segment(data = subset(arrows, !head), aes(x, y, xend = xend, yend = yend), linewidth = 0.25) +
  annotation_raster(ras, xmin = 1, xmax = 37, ymin = 4, ymax = 25) +
  annotate("rect", xmin = 1, xmax = 37, ymin = 4, ymax = 25, fill = NA, colour = "black", linewidth = 0.25) +
  scale_fill_manual(values = FILL, na.value = "white", guide = "none") +
  coord_fixed(xlim = c(0, W), ylim = c(0, H), expand = FALSE) +
  theme_void() + theme(plot.margin = margin(1, 1, 1, 1, "mm"))

save_figure(fig, "figures/fig-01-concept.pdf", width_mm = W_2COL, height_mm = 106)
save_figure(fig, "figures/fig-01-concept.png", width_mm = W_2COL, height_mm = 106, dpi = 300)
