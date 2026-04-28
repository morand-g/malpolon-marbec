library(readr)
library(dplyr)
library(stringr)
library(purrr)
library(progress)
library(modEvA)

inputs_path <- "/marbec-data/RLS-Australia/malpolon/inputs/australia/"
output_path <- "/marbec-data/RLS-Australia/malpolon/outputs/"

# Load data
fulldf <- read_csv(
  file.path(inputs_path, "biomass_lognorm.csv"),
  col_types = cols(
    `22` = col_character(),
    `24` = col_character(),
    `25` = col_character()
  )
)

fulldf <- as.data.frame(fulldf)
rownames(fulldf) <- fulldf$survey_id
fulldf$survey_id <- NULL

species <- colnames(fulldf)[(ncol(fulldf) - 817):(ncol(fulldf)-1)]
groundtruth <- as.data.frame(lapply(fulldf[, species], function(x) as.integer(x > 0)))
rownames(groundtruth) <- rownames(fulldf)


pred_dir <- file.path(output_path, "42_ind_pa_seedtest-2026-03-12_17-19")

for (i in 1:5) {
  
  print(paste("Processing Seed ",i))
  
  tss_dict <- list()
  
  preds_test <- read_csv(file.path(pred_dir, paste0("Seed-", i), "predictions-probs.csv"), col_types = cols())
  preds_test <- as.data.frame(preds_test)
  rownames(preds_test) <- preds_test[[1]]
  preds_test[[1]] <- NULL
  
  for (s in species) {
    targ <- groundtruth[rownames(preds_test), make.names(c(s))]
    tss_dict[s] <- Boyce(obs=targ, pred=preds_test[[s]], plot=FALSE)$Boyce
  }
  
  tss <- data.frame(
    species = names(tss_dict),
    boyce = as.numeric(unlist(tss_dict))
  )
  
  tss <- tss %>%
    arrange(desc(boyce))
  
  # Write to CSV
  write.csv(
    tss,
    file = file.path(pred_dir, paste0("Seed-", i), "boyce.csv"),
    row.names = FALSE
  )
}


