library(readr)
library(dplyr)
library(stringr)
library(purrr)
library(progress)
library(modEvA)

root <- "/marbec-data/RLS-Australia/malpolon/xgb/"
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


# ---- best_f1 function ----
best_f1 <- function(sp, ss, seed) {
  
  sp_safe <- str_replace_all(sp, "/", "-")
  
  file_path <- file.path(
    root, "pa", paste0("Seed-", seed),
    paste0("xgb-preds-", ss),
    paste0("test_", sp_safe, ".csv")
  )
  
  preds_test <- read_csv(file_path, col_types = cols())
  preds_test <- as.data.frame(preds_test)
  rownames(preds_test) <- preds_test[[1]]
  preds_test[[1]] <- NULL

  targ <- groundtruth[rownames(preds_test), make.names(c(sp))]
  
  return(Boyce(obs=targ, pred=preds_test[[1]], plot = FALSE)$Boyce)
}

# ---- main loop ----
for (seed in 1:5) {
  
  print(paste("Processing Seed ", seed))
  
  scoresxgb <- data.frame(row.names = species)
  threshold <- list()
  
  for (ss in seq(0.1, 0.9, by = 0.2)) {
    
    pattern <- sprintf("xgb_val_f1-%.1f-.*\\.csv", ss)
    
    files <- list.files(
      path = file.path(root, "pa", paste0("Seed-", seed)),
      pattern = pattern,
      full.names = TRUE
    )
    
    df <- read_csv(files[1], col_types = cols())
    
    scoresxgb[[sprintf("%.1f", ss)]] <- df$f1
  }
  
  # ---- best subset selection ----
  best_idx <- data.frame(
    ss = apply(scoresxgb, 1, function(x) colnames(scoresxgb)[which.max(x)]),
    stringsAsFactors = FALSE
  )
  
  f1 <- pmap_dbl(
    list(
      sp = rownames(best_idx),
      ss = best_idx$ss
    ),
    function(sp, ss) {
      best_f1(sp, ss, seed)
    }
  )
  
  xgb_f1 <- data.frame(boyce = f1, row.names = rownames(best_idx)) %>%
    arrange(desc(boyce))
  
  write.csv(
    xgb_f1,
    file.path(root, "pa", paste0("Seed-", seed), "boyce.csv"),
    row.names=TRUE
  )
}