library(data.table)
library(stringr)
library(ggplot2)
library(scales)
setwd("./R_figures")
source("util_analysis.R")


# Get a list of all directories

directories <- list.dirs(path = "../models/", full.names = F, recursive = TRUE)
directories

#directories_v4
directories_v4 <- directories[grep("(DeltaTopic|BALSAM)_ep2000_nlv32.*v4$", directories)]
directories_v4
# Apply the function to each directory and combine into one data.table
dt_all_v4 <- rbindlist(lapply(paste0("../models/",directories_v4), read_and_parse))

# Plot

DT <- dt_all_v4
DT[Beta == "unspliced", Beta := "Static Loading"]
DT[Beta == "spliced", Beta := "Dynamic Loading"]
DT[, Col := paste0("topic", Col+1)]
DT[, Weight_Beta := paste(Weight, Beta, sep = "_")]
DT[, Weight_new := ifelse((method == "BALSAM" & Weight == "rho"), "beta",
                                           Weight)]

DT[, Precision := Max/k]

########

dt_summary <- DT[, .(Mean_Precision = mean(Precision),
                    SE_Precision = sd(Precision)/sqrt(.N)), by = .(Weight_new, Beta, pip, Type, k)]
dt_summary[Weight_new %in% c("rho", 'delta'), model := "DeltaTopic",]
dt_summary[Weight_new %in% c('beta'), model := "BALSAM",]
dt_summary[Weight_new %in% c('pca_spliced'), model := "PCA",]
dt_summary[Weight_new %in% c('pca_concat'), model := "PCA-concat",]

# Boxplots
dt_summary_to_plot <- dt_summary[Type == "top" & k != 5000 & !(Weight_new == "rho" & Beta == 'Dynamic Loading') & !(Weight_new == "delta" & Beta == 'Static Loading')]

p <- ggplot(dt_summary_to_plot, aes(x = k, y = Mean_Precision, fill = model, group = model)) +
  facet_grid(Beta ~ .) +
  geom_errorbar(aes(ymin = (Mean_Precision - 1.96 * SE_Precision), ymax = (Mean_Precision + 1.96 * SE_Precision), colour = model), 
                width = 0.1) +
  geom_line(aes(colour = model)) + theme_minimal() +
  labs(y = "Precision in top K genes", x = "K") +
  geom_point(aes(colour = model), size = 4, shape = 18) +
  scale_y_continuous(labels = percent_format()) +
  scale_x_log10()
p
ggsave("benchmark_precision.pdf", p, width = 10, height = 20, units = "in", dpi = 300)


# Plot the nmi
dt_all_nmi <- rbindlist(lapply(paste0("../models/",directories_v4), filename = "nmi_values.txt", read_and_parse))
dt_all_nmi[!model%in% c("PCA-concat", 'PCA'), model:= method,]
setnames(dt_all_nmi, "V1", "nmi")

dt_nmi_summary <- dt_all_nmi[, .(Mean_nmi = mean(nmi),
                    SE_nmi = sd(nmi)/sqrt(.N)), by = .(model)]
dt_nmi_summary


p_nmi <- ggplot(dt_nmi_summary, aes(x = model, y = Mean_nmi, fill = model)) +
  geom_errorbar(aes(ymin = (Mean_nmi - 1.96 * SE_nmi), ymax = (Mean_nmi + 1.96 * SE_nmi), colour = model), 
                width = 0.1) +
  geom_line(aes(colour = model)) + theme_minimal() +
  labs(y = "NMI", x = "model") +
  geom_point(aes(colour = model), size = 4, shape = 18) 
p_nmi

ggsave("benchmark_nmi.pdf", p_nmi, width = 10, height = 20, units = "in", dpi = 300)